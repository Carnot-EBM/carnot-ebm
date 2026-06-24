"""Tests for Exp 4694 proposal-coverage CI-gate.

Spec refs: REQ-ARC-WMTE-4694,
SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE,
SCENARIO-ARC-WMTE-4694-HONEST-FIRSTWIN,
SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE-FLOOR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _winning_trajectory() -> list[dict[str, Any]]:
    return [
        {"action": 6, "data": {"x": 37, "y": 44}},
        {"action": 2, "data": None},
        {"action": 5, "data": {"x": 10, "y": 11}},
    ]


def _trace(
    signature: str,
    *,
    trajectory: list[dict[str, Any]] | None = None,
    missing_index: int | None = None,
) -> dict[str, Any]:
    winner = list(trajectory or _winning_trajectory())
    proposal_steps = []
    for index, step in enumerate(winner):
        proposed = {"action": 1, "data": None} if index == missing_index else step
        proposal_steps.append(
            {
                "step_index": index,
                "prefix": winner[:index],
                "action_proposals": [proposed, {"action": 9, "data": {"x": index}}],
            }
        )
    return {
        "variant_signature": signature,
        "exploration_trace": {"proposal_steps": proposal_steps},
        "winning_l1_trajectory": winner,
    }


def test_req_arc_wmte_4694_spec_declares_proposal_coverage_contract() -> None:
    """REQ-ARC-WMTE-4694: OpenSpec anchors all guard fields and principles."""

    from carnot import experiment_4694_proposal_coverage_cigate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4694" in spec
    assert "SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE" in spec
    assert "SCENARIO-ARC-WMTE-4694-HONEST-FIRSTWIN" in spec
    assert "SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE-FLOOR" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4694_proposal_coverage_present_and_absent() -> None:
    """SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE: full trajectory proposed is coverage=1."""

    from carnot import experiment_4694_proposal_coverage_cigate as mod

    winner = _winning_trajectory()
    present = mod.l1_proposal_coverage(_trace("lp85~color01"), winner)
    absent = mod.l1_proposal_coverage(_trace("lp85~color01", missing_index=1), winner)

    assert present["winning_trajectory_reached"] is True
    assert present["proposal_coverage"] == pytest.approx(1.0)
    assert present["proposed_step_count"] == 3
    assert present["trajectory_length"] == 3
    assert absent["winning_trajectory_reached"] is False
    assert absent["proposal_coverage"] == pytest.approx(0.0)
    assert absent["first_missed_step_index"] == 1


def test_scenario_arc_wmte_4694_coverage_up_vs_flat_baseline_gate_passes() -> None:
    """SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE: method coverage must beat flat."""

    from carnot import experiment_4694_proposal_coverage_cigate as mod

    method = [_trace("lp85~color01"), _trace("sc25~color01", missing_index=2)]
    flat = [
        _trace("lp85~color01", missing_index=0),
        _trace("sc25~color01", missing_index=2),
    ]

    gate = mod.validate_proposal_coverage_gate(method, flat)
    collapsed = mod.validate_proposal_coverage_gate(flat, flat)

    assert gate["passed"] is True
    assert gate["method"]["coverage_rate"] == pytest.approx(0.5)
    assert gate["baseline"]["coverage_rate"] == pytest.approx(0.0)
    assert gate["coverage_delta"] == pytest.approx(0.5)
    assert collapsed["passed"] is False
    assert "proposal_coverage_not_above_baseline" in collapsed["errors"]
    with pytest.raises(mod.GateFailure, match="proposal_coverage_not_above_baseline"):
        mod.assert_proposal_coverage_gate(collapsed)


def test_scenario_arc_wmte_4694_honest_firstwin_standard_passes_permissive_fails() -> None:
    """SCENARIO-ARC-WMTE-4694-HONEST-FIRSTWIN: permissive harnesses are flagged."""

    from carnot import experiment_4694_proposal_coverage_cigate as mod

    standard_measurement = {
        "first_win_count": 1,
        "first_win_rate": 0.04,
        "variant_attempts_count": 25,
        "variant_signatures": list(mod.STANDARD_VARIANT_SIGNATURES),
    }
    standard = mod.validate_honest_firstwin_measurement(
        standard_measurement,
        config={"action_budget": 200, "policy_mode": "4676_explore_budget_200"},
    )
    permissive = mod.validate_honest_firstwin_measurement(
        {
            "first_win_count": 1,
            "first_win_rate": 1.0,
            "variant_attempts_count": 1,
            "variant_signatures": ["lp85~color01"],
        },
        config={"action_budget": None, "policy_mode": "4676_explore_budget_200"},
    )
    below_floor = mod.validate_honest_firstwin_measurement(
        dict(standard_measurement, first_win_count=0, first_win_rate=0.0),
        config={"action_budget": 200, "policy_mode": "4676_explore_budget_200"},
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


def test_scenario_arc_wmte_4694_proposal_coverage_floor_flags_regression() -> None:
    """SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE-FLOOR: below-floor coverage fails."""

    from carnot import experiment_4694_proposal_coverage_cigate as mod

    regressed = mod.validate_proposal_coverage_floor({"coverage_rate": 0.03}, floor=0.04)
    honest = mod.validate_proposal_coverage_floor({"coverage_rate": 0.04}, floor=0.04)

    assert regressed["passed"] is False
    assert "proposal_coverage_below_floor" in regressed["errors"]
    assert honest["passed"] is True
    with pytest.raises(mod.GateFailure, match="proposal_coverage_below_floor"):
        mod.assert_proposal_coverage_floor(regressed)


def test_req_arc_wmte_4694_artifact_schema_and_run_write_terminal_json(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4694: terminal artifact is checksummed and schema-validated."""

    from carnot import experiment_4694_proposal_coverage_cigate as mod

    method = [_trace("lp85~color01")]
    flat = [_trace("lp85~color01", missing_index=0)]
    firstwin = {
        "first_win_count": 1,
        "first_win_rate": 0.04,
        "variant_attempts_count": 25,
        "variant_signatures": list(mod.STANDARD_VARIANT_SIGNATURES),
    }

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        proposal_coverage_metric_added=mod.assert_proposal_coverage_gate(
            mod.validate_proposal_coverage_gate(method, flat)
        ),
        honest_firstwin_floor_added=mod.assert_honest_firstwin_measurement(
            mod.validate_honest_firstwin_measurement(
                firstwin,
                config={"action_budget": 200, "policy_mode": "4676_explore_budget_200"},
            )
        ),
        proposal_coverage_floor_cigate_added=mod.assert_proposal_coverage_floor(
            mod.validate_proposal_coverage_floor({"coverage_rate": 0.04}, floor=0.04)
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
        firstwin_config={"action_budget": 200, "policy_mode": "4676_explore_budget_200"},
        proposal_coverage_floor=0.04,
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
        "success: proposal_coverage_cigate_plus_honest_firstwin_floor_shipped_tests_green"
    )
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
    assert loaded == written
    assert written["proposal_coverage_metric_added"]["passed"] is True
    assert written["honest_firstwin_floor_added"]["passed"] is True
    assert written["proposal_coverage_floor_cigate_added"]["passed"] is True
    assert blocked["honest_verdict"] == "blocked_offline_arcade"


def test_req_arc_wmte_4694_helper_branches_and_schema_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4694: defensive branches stay deterministic and covered."""

    from carnot import experiment_4694_proposal_coverage_cigate as mod

    winner = _winning_trajectory()
    label_winner = [json.dumps(step, sort_keys=True, separators=(",", ":")) for step in winner]
    nested = {
        "trace": {
            "proposal_distribution": [
                [json.dumps(winner[0], sort_keys=True, separators=(",", ":"))],
                [{"action": 2, "data": None}],
                [{"candidate": winner[2]}],
            ]
        }
    }
    one_step = {"proposed_actions": {"a": json.dumps(winner[0], sort_keys=True)}}
    raw_label = {"proposals_by_depth": {"0": ["raw-action"]}}
    wrapped_single = {"proposal_steps": [{"proposals": [[winner[0]]]}]}
    scalar_pool = {"proposal_steps": [{"step_index": 0, "proposals": "raw-action"}]}
    direct_action_record = {"proposal_steps": [{"action": 7, "data": None}]}
    sequence_record = {"proposal_steps": [[{"action": 8, "data": None}]]}
    no_depth_fallback = {"proposal_steps": [{"proposals": [{"action": 9, "data": None}]}]}
    mismatched_prefix_then_depth = {
        "proposal_steps": [
            {
                "prefix": [{"action": 0, "data": None}],
                "step_index": 0,
                "action_proposals": [{"action": 1, "data": None}],
            },
            {"step_index": 0, "action_proposals": [{"action": 3, "data": None}]},
        ]
    }
    distribution_mapping = {"proposal_distribution": {"0": [{"action": 4, "data": None}]}}
    distribution_row_mapping = {
        "proposal_distribution": [
            {"action_proposals": [{"action": 5, "data": None}]},
        ]
    }
    records_metric = mod.measure_proposal_coverage({"records": [_trace("lp85~color01")]})
    missing_winner_metric = mod.measure_proposal_coverage(
        [{"variant_signature": "lp85~color01", "proposal_steps": []}]
    )
    missing_gate = mod.validate_proposal_coverage_gate([], [])
    mismatch_gate = mod.validate_proposal_coverage_gate(
        [_trace("lp85~color01")],
        [_trace("sc25~color01", missing_index=0)],
    )

    assert mod.l1_proposal_coverage(nested, label_winner)["winning_trajectory_reached"] is True
    assert mod.l1_proposal_coverage(one_step, [winner[0]])["winning_trajectory_reached"] is True
    assert mod.l1_proposal_coverage(raw_label, ["raw-action"])["winning_trajectory_reached"] is True
    assert mod.l1_proposal_coverage(raw_label, "raw-action")["winning_trajectory_reached"] is True
    assert mod.l1_proposal_coverage(wrapped_single, [[winner[0]]])[
        "winning_trajectory_reached"
    ] is True
    assert mod.l1_proposal_coverage(scalar_pool, ["raw-action"])[
        "winning_trajectory_reached"
    ] is True
    assert mod.l1_proposal_coverage(direct_action_record, {"action": "7", "data": None})[
        "winning_trajectory_reached"
    ] is True
    assert mod.l1_proposal_coverage(sequence_record, [{"action": 8, "data": None}])[
        "winning_trajectory_reached"
    ] is True
    assert mod.l1_proposal_coverage(no_depth_fallback, {"trajectory": [{"action": 9, "data": None}]})[
        "winning_trajectory_reached"
    ] is True
    assert mod.l1_proposal_coverage(mismatched_prefix_then_depth, [{"action": 3, "data": None}])[
        "winning_trajectory_reached"
    ] is True
    assert mod.l1_proposal_coverage(distribution_mapping, [{"action": 4, "data": None}])[
        "winning_trajectory_reached"
    ] is True
    assert mod.l1_proposal_coverage(distribution_row_mapping, [{"action": 5, "data": None}])[
        "winning_trajectory_reached"
    ] is True
    assert mod.l1_proposal_coverage({}, winner)["trajectory_length"] == 3
    assert mod.l1_proposal_coverage({}, {"unknown": "shape"})["trajectory_length"] == 0
    assert mod._proposal_steps_for_trajectory(winner, missing_index=1)[1]["action_proposals"] == [
        {"action": 1, "data": None}
    ]
    assert records_metric["attempted_count"] == 1
    assert missing_winner_metric["coverage_rate"] == pytest.approx(0.0)
    assert "method_proposal_traces_missing" in missing_gate["errors"]
    assert "flat_baseline_proposal_traces_missing" in missing_gate["errors"]
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
    assert "proposal_coverage_below_floor" in mod.validate_proposal_coverage_floor(
        {"coverage_rate": "bad"}, floor=0.01
    )["errors"]
    assert mod.validate_proposal_coverage_floor(
        {"method": {"coverage_rate": 0.05}}, floor=0.04
    )["passed"] is True
    assert mod.validate_proposal_coverage_floor(
        {"records": [_trace("lp85~color01")]}, floor=1.0
    )["passed"] is True

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposal_coverage_metric_added={
            "passed": True,
            "errors": [],
            "method": {"coverage_rate": 1.0},
            "baseline": {"coverage_rate": 0.0},
        },
        honest_firstwin_floor_added={"passed": True, "errors": []},
        proposal_coverage_floor_cigate_added={"passed": True, "errors": []},
        tests_added={"passed": True},
        duration_s=1.0,
    )
    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "not_terminal",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "proposal_coverage_metric_added": [],
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
    assert "proposal_coverage_metric_added" in errors
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

    method = [_trace("lp85~color01")]
    flat = [_trace("lp85~color01", missing_index=0)]
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
