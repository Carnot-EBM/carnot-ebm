"""Tests for Exp 4682 candidate-generation coverage CI-gate.

Spec refs: REQ-ARC-WMTE-4682,
SCENARIO-ARC-WMTE-4682-COVERAGE-METRIC,
SCENARIO-ARC-WMTE-4682-HONEST-FIRSTWIN,
SCENARIO-ARC-WMTE-4682-COVERAGE-FLOOR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _trace(
    signature: str,
    *,
    pool: list[Any],
    winner: Any,
) -> dict[str, Any]:
    return {
        "variant_signature": signature,
        "search_trace": {"generated_candidates": pool},
        "winning_plan": winner,
    }


def test_req_arc_wmte_4682_spec_declares_generation_coverage_contract() -> None:
    """REQ-ARC-WMTE-4682: OpenSpec anchors all guard fields and principles."""

    from carnot import experiment_4682_generation_coverage_cigate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4682" in spec
    assert "SCENARIO-ARC-WMTE-4682-COVERAGE-METRIC" in spec
    assert "SCENARIO-ARC-WMTE-4682-HONEST-FIRSTWIN" in spec
    assert "SCENARIO-ARC-WMTE-4682-COVERAGE-FLOOR" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4682_candidate_generation_coverage_present_and_absent() -> None:
    """SCENARIO-ARC-WMTE-4682-COVERAGE-METRIC: winner in pool is coverage=1."""

    from carnot import experiment_4682_generation_coverage_cigate as mod

    winner = [{"action": 6, "data": {"x": 37, "y": 44}}]
    present = mod.candidate_generation_coverage(
        {"generated_candidates": [{"plan": winner}, {"plan": [{"action": 1}]}]},
        winner,
    )
    absent = mod.candidate_generation_coverage(
        {"generated_candidates": [{"plan": [{"action": 1}]}]},
        winner,
    )

    assert present["winner_present"] is True
    assert present["coverage"] == pytest.approx(1.0)
    assert present["candidate_count"] == 2
    assert absent["winner_present"] is False
    assert absent["coverage"] == pytest.approx(0.0)
    assert absent["candidate_count"] == 1


def test_scenario_arc_wmte_4682_coverage_up_vs_flat_baseline_gate_passes() -> None:
    """SCENARIO-ARC-WMTE-4682-COVERAGE-METRIC: method coverage must beat flat."""

    from carnot import experiment_4682_generation_coverage_cigate as mod

    winner = [{"action": 6, "data": {"x": 37, "y": 44}}]
    method = [
        _trace("lp85~color01", pool=[{"plan": winner}], winner=winner),
        _trace("sc25~color01", pool=[{"plan": [{"action": 1}]}], winner=winner),
    ]
    flat = [
        _trace("lp85~color01", pool=[{"plan": [{"action": 1}]}], winner=winner),
        _trace("sc25~color01", pool=[{"plan": [{"action": 1}]}], winner=winner),
    ]

    gate = mod.validate_candidate_generation_coverage_gate(method, flat)
    collapsed = mod.validate_candidate_generation_coverage_gate(flat, flat)

    assert gate["passed"] is True
    assert gate["method"]["coverage_rate"] == pytest.approx(0.5)
    assert gate["baseline"]["coverage_rate"] == pytest.approx(0.0)
    assert gate["coverage_delta"] == pytest.approx(0.5)
    assert collapsed["passed"] is False
    assert "candidate_generation_coverage_not_above_baseline" in collapsed["errors"]
    with pytest.raises(mod.GateFailure, match="candidate_generation_coverage_not_above_baseline"):
        mod.assert_candidate_generation_coverage_gate(collapsed)


def test_scenario_arc_wmte_4682_honest_firstwin_standard_passes_permissive_fails() -> None:
    """SCENARIO-ARC-WMTE-4682-HONEST-FIRSTWIN: permissive harnesses are flagged."""

    from carnot import experiment_4682_generation_coverage_cigate as mod

    standard_measurement = {
        "first_win_count": 1,
        "first_win_rate": 0.04,
        "variant_attempts_count": 25,
        "variant_signatures": list(mod.STANDARD_VARIANT_SIGNATURES),
    }
    standard = mod.validate_honest_firstwin_measurement(
        standard_measurement,
        config={"action_budget": 200, "policy_mode": "value_routed"},
    )
    permissive = mod.validate_honest_firstwin_measurement(
        {
            "first_win_count": 1,
            "first_win_rate": 1.0,
            "variant_attempts_count": 1,
            "variant_signatures": ["lp85~color01"],
        },
        config={"action_budget": None, "policy_mode": "value_routed"},
    )
    below_floor = mod.validate_honest_firstwin_measurement(
        dict(standard_measurement, first_win_count=0, first_win_rate=0.0),
        config={"action_budget": 200, "policy_mode": "value_routed"},
    )

    assert standard["passed"] is True
    assert standard["measured"]["first_win_rate"] == pytest.approx(0.04)
    assert permissive["passed"] is False
    assert "variant_set_not_standard" in permissive["errors"]
    assert "degenerate_easy_variant_subset" in permissive["errors"]
    assert "action_budget_unbounded" in permissive["errors"]
    assert "first_win_rate_below_floor" in below_floor["errors"]
    with pytest.raises(mod.GateFailure, match="variant_set_not_standard"):
        mod.assert_honest_firstwin_measurement(permissive)


def test_scenario_arc_wmte_4682_generation_coverage_floor_flags_regression() -> None:
    """SCENARIO-ARC-WMTE-4682-COVERAGE-FLOOR: below-floor coverage fails."""

    from carnot import experiment_4682_generation_coverage_cigate as mod

    regressed = mod.validate_generation_coverage_floor({"coverage_rate": 0.03}, floor=0.04)
    honest = mod.validate_generation_coverage_floor({"coverage_rate": 0.04}, floor=0.04)

    assert regressed["passed"] is False
    assert "candidate_generation_coverage_below_floor" in regressed["errors"]
    assert honest["passed"] is True
    with pytest.raises(mod.GateFailure, match="candidate_generation_coverage_below_floor"):
        mod.assert_generation_coverage_floor(regressed)


def test_req_arc_wmte_4682_artifact_schema_and_run_write_terminal_json(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4682: terminal artifact is checksummed and schema-validated."""

    from carnot import experiment_4682_generation_coverage_cigate as mod

    winner = [{"action": 6, "data": {"x": 37, "y": 44}}]
    method = [_trace("lp85~color01", pool=[{"plan": winner}], winner=winner)]
    flat = [_trace("lp85~color01", pool=[{"plan": [{"action": 1}]}], winner=winner)]
    firstwin = {
        "first_win_count": 1,
        "first_win_rate": 0.04,
        "variant_attempts_count": 25,
        "variant_signatures": list(mod.STANDARD_VARIANT_SIGNATURES),
    }

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        coverage_metric_added=mod.assert_candidate_generation_coverage_gate(
            mod.validate_candidate_generation_coverage_gate(method, flat)
        ),
        honest_firstwin_floor_added=mod.assert_honest_firstwin_measurement(
            mod.validate_honest_firstwin_measurement(
                firstwin,
                config={"action_budget": 200, "policy_mode": "value_routed"},
            )
        ),
        coverage_floor_cigate_added=mod.assert_generation_coverage_floor(
            mod.validate_generation_coverage_floor({"coverage_rate": 0.04}, floor=0.04)
        ),
        tests_added={"passed": True, "test_file": __file__},
        duration_s=1.0,
    )
    written = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": True, "offline_arcade": True},
        method_traces=method,
        flat_baseline_traces=flat,
        firstwin_measurement=firstwin,
        firstwin_config={"action_budget": 200, "policy_mode": "value_routed"},
        coverage_floor=0.04,
        duration_s=1.0,
        write=True,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    blocked = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.0,
        write=False,
    )

    assert artifact["honest_verdict"] == (
        "success: generation_coverage_cigate_plus_honest_firstwin_floor_shipped_tests_green"
    )
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
    assert loaded == written
    assert written["coverage_metric_added"]["passed"] is True
    assert written["honest_firstwin_floor_added"]["passed"] is True
    assert written["coverage_floor_cigate_added"]["passed"] is True
    assert blocked["honest_verdict"] == "blocked_offline_arcade"


def test_req_arc_wmte_4682_helper_branches_and_schema_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4682: defensive branches stay deterministic and covered."""

    from carnot import experiment_4682_generation_coverage_cigate as mod

    winner = {"action": 6, "data": {"x": 1, "y": 2}}
    assert mod.candidate_generation_coverage(
        {"search_trace": {"candidate_pool": {"a": json.dumps(winner)}}},
        [winner],
    )["winner_present"] is True
    assert mod.candidate_generation_coverage(
        {"generated_candidates": ["not-json"]},
        "not-json",
    )["winner_present"] is True
    assert mod.candidate_generation_coverage({}, winner)["candidate_count"] == 0

    records_metric = mod.measure_candidate_generation_coverage(
        {"traces": [_trace("lp85~color01", pool=[{"action": 1}], winner=winner)]}
    )
    missing_winner_metric = mod.measure_candidate_generation_coverage(
        [{"variant_signature": "lp85~color01", "generated_candidates": [{"action": 1}]}]
    )
    missing_gate = mod.validate_candidate_generation_coverage_gate([], [])
    mismatch_gate = mod.validate_candidate_generation_coverage_gate(
        [_trace("lp85~color01", pool=[winner], winner=winner)],
        [_trace("sc25~color01", pool=[{"action": 1}], winner=winner)],
    )

    assert records_metric["attempted_count"] == 1
    assert missing_winner_metric["coverage_rate"] == pytest.approx(0.0)
    assert "method_candidate_traces_missing" in missing_gate["errors"]
    assert "flat_baseline_candidate_traces_missing" in missing_gate["errors"]
    assert "matched_variant_signatures_required" in mismatch_gate["errors"]

    attempts = [
        {"variant_signature": sig, "first_win": index == 0}
        for index, sig in enumerate(mod.STANDARD_VARIANT_SIGNATURES)
    ]
    computed_firstwin = mod.validate_honest_firstwin_measurement(
        {"variant_attempts": attempts},
        config={"action_budget": 200},
    )
    bad_budget = mod.validate_honest_firstwin_measurement(
        {
            "first_win_count": 1,
            "first_win_rate": 0.04,
            "variant_attempts_count": 25,
            "variant_signatures": list(mod.STANDARD_VARIANT_SIGNATURES),
        },
        config={"action_budget": "bad"},
    )
    missing_budget = mod.validate_honest_firstwin_measurement(
        {
            "first_win_count": 1,
            "first_win_rate": 0.04,
            "variant_attempts_count": 25,
            "variant_signatures": list(mod.STANDARD_VARIANT_SIGNATURES),
        }
    )

    assert computed_firstwin["passed"] is True
    assert computed_firstwin["measured"]["first_win_rate"] == pytest.approx(0.04)
    assert "action_budget_not_standard" in bad_budget["errors"]
    assert "action_budget_unbounded" in missing_budget["errors"]
    assert "candidate_generation_coverage_below_floor" in mod.validate_generation_coverage_floor(
        {"coverage_rate": "bad"}, floor=0.01
    )["errors"]
    assert mod.validate_generation_coverage_floor(
        {"method": {"coverage_rate": 0.05}}, floor=0.04
    )["passed"] is True
    assert mod.validate_generation_coverage_floor(
        {"traces": [_trace("lp85~color01", pool=[winner], winner=winner)]}, floor=1.0
    )["passed"] is True

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        coverage_metric_added={
            "passed": True,
            "errors": [],
            "method": {"coverage_rate": 1.0},
            "baseline": {"coverage_rate": 0.0},
        },
        honest_firstwin_floor_added={"passed": True, "errors": []},
        coverage_floor_cigate_added={"passed": True, "errors": []},
        tests_added={"passed": True},
        duration_s=1.0,
    )
    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "not_terminal",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "coverage_metric_added": [],
            "field_principles": [],
            "reproducibility_checksum": "bad",
        }
    )
    principle_missing = dict(artifact)
    principles = dict(artifact["field_principles"])
    principles.pop("tests_added")
    principle_missing["field_principles"] = principles

    errors = mod.artifact_schema_errors(bad)
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "verifier_is_oracle_false" in errors
    assert "coverage_metric_added" in errors
    assert "field_principles" in errors
    assert "reproducibility_checksum" in errors
    assert "field_principles.tests_added" in mod.artifact_schema_errors(principle_missing)

    blocked_written = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.0,
        write=True,
    )
    assert blocked_written["honest_verdict"] == "blocked_offline_arcade"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    method = [_trace("lp85~color01", pool=[winner], winner=winner)]
    flat = [_trace("lp85~color01", pool=[{"action": 1}], winner=winner)]
    firstwin = {
        "first_win_count": 1,
        "first_win_rate": 0.04,
        "variant_attempts_count": 25,
        "variant_signatures": list(mod.STANDARD_VARIANT_SIGNATURES),
    }
    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced_schema_error"])
    with pytest.raises(mod.GateFailure, match="forced_schema_error"):
        mod.run(
            root=tmp_path,
            preconditions_checked={"ok": True},
            method_traces=method,
            flat_baseline_traces=flat,
            firstwin_measurement=firstwin,
            firstwin_config={"action_budget": 200},
            duration_s=1.0,
            write=False,
        )
