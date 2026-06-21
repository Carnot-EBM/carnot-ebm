"""Tests for Exp 4524 CORE L1->L2 barrier diagnosis.

Spec refs: REQ-ARC-WMTE-4524, SCENARIO-ARC-WMTE-4524.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4524_reach_deeper_levels as exp4524


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions(ok: bool = True) -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": ok,
        "spec_has_req_4524": True,
        "ok": ok,
    }


def _measurement(
    lever: str,
    *,
    efficiency: float,
    lp85_level: int = 1,
    lp85_diag: dict[str, object] | None = None,
    m0r0_level: int = 1,
    sp80_level: int = 1,
) -> dict[str, object]:
    diag = {
        "stopped_reason": "explored_out",
        "max_depth": 45,
        "max_depth_reached": 44,
        "depth_cap_frontier_nodes": 0,
        "candidate_count_at_last_l1": 0,
        "known_l2_transition_in_salience": True,
        "l2_win_condition_differs_from_l1": True,
        "world_model_induction_invoked": False,
        "actionable_next_step": "route L1 plateau states into a level-conditioned lp85 goal predicate.",
    }
    if lp85_diag:
        diag.update(lp85_diag)
    return {
        "lever": lever,
        "description": f"{lever} measurement",
        "core_efficiency": efficiency,
        "per_game": [
            {"game": "lp85", "best_level": lp85_level, "diagnostics": diag},
            {"game": "m0r0", "best_level": m0r0_level, "diagnostics": {"stopped_reason": "target_reached"}},
            {"game": "sp80", "best_level": sp80_level, "diagnostics": {"stopped_reason": "target_reached"}},
        ],
    }


def test_req_arc_wmte_4524_spec_declares_required_artifact_contract() -> None:
    """REQ-ARC-WMTE-4524: OpenSpec anchors the reach-deeper-levels artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4524" in spec
    assert "SCENARIO-ARC-WMTE-4524" in spec
    assert exp4524.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4524.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4524_honest_null_records_concrete_barrier(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4524: no L2 writes a concrete L1->L2 barrier diagnosis."""

    measurements = [
        _measurement("control_max_depth_45", efficiency=exp4524.CORE_EFFICIENCY_BASELINE),
        _measurement(
            "deeper_search_max_depth_90",
            efficiency=exp4524.CORE_EFFICIENCY_BASELINE,
            lp85_diag={"max_depth": 90, "max_depth_reached": 57},
        ),
        _measurement(
            "world_model_dsl_induction",
            efficiency=exp4524.CORE_EFFICIENCY_BASELINE,
            lp85_diag={"world_model_induction_invoked": False},
        ),
        _measurement(
            "energy_verifier_frontier_routing",
            efficiency=exp4524.CORE_EFFICIENCY_BASELINE,
            lp85_diag={"energy_signal_available": False},
        ),
    ]
    artifact = exp4524.build_artifact(
        preconditions_checked=_preconditions(),
        measurements=measurements,
        offline_reproduction={},
        random_seed=4524,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == (
        "complete: l1_l2_barrier_diagnosed_new_win_condition_honest_null"
    )
    assert artifact["core_efficiency_baseline"] == exp4524.CORE_EFFICIENCY_BASELINE
    assert artifact["core_efficiency_best"] == exp4524.CORE_EFFICIENCY_BASELINE
    assert artifact["efficiency_delta"] == 0.0
    assert "no lever reached a deeper CORE level" in artifact["null_delta_methodology_note"]
    assert artifact["offline_reproduced"] is False
    assert artifact["barrier_diagnosis"]["root_cause"] == "new_win_condition"
    assert artifact["barrier_diagnosis"]["target_game"] == "lp85"
    assert artifact["barrier_diagnosis"]["depth_cap_likely"] is False
    assert artifact["barrier_diagnosis"]["missing_salience_likely"] is False
    assert artifact["barrier_diagnosis"]["new_win_condition_likely"] is True
    assert artifact["barrier_diagnosis"]["induction_not_engaged"] is True
    assert artifact["deepest_level_reached_per_core_game"]["deeper_search_max_depth_90"]["lp85"] == 1
    assert [row["lever"] for row in artifact["levers_tried"]] == [
        "control_max_depth_45",
        "deeper_search_max_depth_90",
        "world_model_dsl_induction",
        "energy_verifier_frontier_routing",
    ]
    assert exp4524.artifact_schema_errors(artifact) == []

    out = exp4524.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact

    changed = dict(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert any("checksum" in error for error in exp4524.artifact_schema_errors(changed))


def test_req_arc_wmte_4524_success_requires_l2_efficiency_gain_and_reproduction() -> None:
    """REQ-ARC-WMTE-4524: reaching L2 only counts with efficiency gain and reproduction."""

    control = _measurement("control_max_depth_45", efficiency=exp4524.CORE_EFFICIENCY_BASELINE)
    candidate = _measurement("energy_verifier_frontier_routing", efficiency=7.25, lp85_level=2)

    artifact = exp4524.build_artifact(
        preconditions_checked=_preconditions(),
        measurements=[control, candidate],
        offline_reproduction={"game": "lp85", "reached_level": 2, "reproduced": True},
        random_seed=4524,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "success: lp85_reached_L2_core_efficiency_7.2500_above_2.0074"
    assert artifact["core_efficiency_best"] == 7.25
    assert artifact["efficiency_delta"] == pytest.approx(5.2426)
    assert "null_delta_methodology_note" not in artifact
    assert artifact["offline_reproduced"] is True
    assert artifact["barrier_diagnosis"]["root_cause"] == "resolved_l2_reached"
    assert exp4524.artifact_schema_errors(artifact) == []

    no_repro = exp4524.build_artifact(
        preconditions_checked=_preconditions(),
        measurements=[control, candidate],
        offline_reproduction={"game": "lp85", "reached_level": 2, "reproduced": False},
        random_seed=4524,
        duration_s=0.5,
    )
    assert no_repro["honest_verdict"] == (
        "complete: l1_l2_barrier_diagnosed_new_win_condition_honest_null"
    )
    assert no_repro["offline_reproduced"] is False

    lost_core = _measurement("energy_verifier_frontier_routing", efficiency=7.25, lp85_level=2, m0r0_level=0)
    regressed = exp4524.build_artifact(
        preconditions_checked=_preconditions(),
        measurements=[control, lost_core],
        offline_reproduction={"game": "lp85", "reached_level": 2, "reproduced": True},
        random_seed=4524,
        duration_s=0.5,
    )
    assert regressed["honest_verdict"] == (
        "complete: l1_l2_barrier_diagnosed_core_level_regression_honest_null"
    )


def test_scenario_arc_wmte_4524_run_writes_injected_measurements(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4524: injected lever measurements write stable JSON."""

    calls: list[str] = []

    def fake_measure(lever: exp4524.LeverConfig) -> dict[str, object]:
        calls.append(lever.name)
        return _measurement(lever.name, efficiency=exp4524.CORE_EFFICIENCY_BASELINE)

    artifact = exp4524.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        measure_lever=fake_measure,
        offline_reproduction_runner=lambda measurement: {},
        now=lambda: 1.0,
    )

    assert calls == [lever.name for lever in exp4524.LEVER_CONFIGS]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["result_path"] == exp4524.RESULT_RELATIVE_PATH
    assert json.loads((tmp_path / exp4524.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4524_missing_offline_arcade_blocks_without_fabrication(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4524: failed preconditions emit blocked_<resource> and no win."""

    artifact = exp4524.run(
        root=tmp_path,
        preconditions_checked=_preconditions(ok=False),
        measure_lever=lambda lever: _measurement(lever.name, efficiency=99.0, lp85_level=2),
        offline_reproduction_runner=lambda measurement: {
            "game": "lp85",
            "reached_level": 2,
            "reproduced": True,
        },
        now=lambda: 1.0,
    )

    assert artifact["honest_verdict"] == "blocked_offline_arcade_import_smoke"
    assert artifact["core_efficiency_best"] == exp4524.CORE_EFFICIENCY_BASELINE
    assert artifact["offline_reproduced"] is False
    assert artifact["levers_tried"] == []
    assert exp4524.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4524_classifier_and_schema_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4524: barrier roots and schema failures are explicit."""

    control = _measurement("control_max_depth_45", efficiency=exp4524.CORE_EFFICIENCY_BASELINE)

    assert exp4524._level_by_game({"best_level_by_game": {"lp85": 1}}) == {"lp85": 1}
    assert exp4524._control_measurement([_measurement("other", efficiency=0.0)])["lever"] == "other"
    assert exp4524._target_diagnostics([{"lever": "empty", "per_game": []}]) == []

    depth = _measurement(
        "control_max_depth_45",
        efficiency=0.0,
        lp85_diag={"depth_cap_frontier_nodes": 2, "l2_win_condition_differs_from_l1": False},
    )
    missing = _measurement(
        "control_max_depth_45",
        efficiency=0.0,
        lp85_diag={
            "known_l2_transition_in_salience": False,
            "l2_win_condition_differs_from_l1": False,
        },
    )
    induction = _measurement(
        "world_model_dsl_induction",
        efficiency=0.0,
        lp85_diag={
            "known_l2_transition_in_salience": True,
            "l2_win_condition_differs_from_l1": False,
            "world_model_induction_invoked": False,
        },
    )
    budget = _measurement(
        "control_max_depth_45",
        efficiency=0.0,
        lp85_diag={
            "stopped_reason": "budget_exhausted",
            "known_l2_transition_in_salience": True,
            "l2_win_condition_differs_from_l1": False,
            "world_model_induction_invoked": True,
        },
    )
    explored = _measurement(
        "control_max_depth_45",
        efficiency=0.0,
        lp85_diag={
            "known_l2_transition_in_salience": True,
            "l2_win_condition_differs_from_l1": False,
            "world_model_induction_invoked": True,
        },
    )
    for expected, measurement in (
        ("depth_cap", depth),
        ("missing_mechanic", missing),
        ("induction_not_engaged", induction),
        ("budget_exhausted", budget),
        ("explored_out", explored),
    ):
        diagnosis = exp4524.diagnose_barrier(
            measurements=[measurement],
            best_measurement=measurement,
            offline_reproduced=False,
            core_level_regressions=[],
        )
        assert diagnosis["root_cause"] == expected

    success = exp4524.build_artifact(
        preconditions_checked=_preconditions(),
        measurements=[control, _measurement("energy_verifier_frontier_routing", efficiency=7.25, lp85_level=2)],
        offline_reproduction={"game": "lp85", "reached_level": 2, "reproduced": True},
        random_seed=4524,
        duration_s=0.5,
    )
    mutations = [
        (lambda item: item.__setitem__("field_principles", {}), "field_principles"),
        (lambda item: item.__setitem__("core_efficiency_baseline", 0.0), "core_efficiency_baseline"),
        (lambda item: item.__setitem__("leaderboard_submission", True), "leaderboard_submission"),
        (lambda item: item.__setitem__("preconditions_checked", None), "preconditions_checked"),
        (
            lambda item: item.__setitem__(
                "preconditions_checked", {"offline_arcade_import_smoke": False}
            ),
            "offline_arcade_import_smoke",
        ),
        (lambda item: item.__setitem__("barrier_diagnosis", None), "barrier_diagnosis"),
        (
            lambda item: item.__setitem__(
                "barrier_diagnosis", {"root_cause": "not_a_root"}
            ),
            "root_cause",
        ),
        (
            lambda item: item.__setitem__("deepest_level_reached_per_core_game", None),
            "deepest_level_reached_per_core_game",
        ),
        (lambda item: item.__setitem__("levers_tried", None), "levers_tried"),
        (lambda item: item.__setitem__("offline_reproduced", False), "offline_reproduced"),
        (lambda item: item.__setitem__("core_efficiency_best", 2.0), "core_efficiency_best"),
        (lambda item: item.__setitem__("no_core_game_loses_level", False), "CORE"),
        (
            lambda item: item.__setitem__(
                "deepest_level_reached_per_core_game", {"x": {"lp85": 1}}
            ),
            "reaches L2",
        ),
        (lambda item: item.__setitem__("reproducibility_checksum", "bad"), "sha256"),
    ]
    for mutate, expected in mutations:
        changed = dict(success)
        mutate(changed)
        assert any(expected in error for error in exp4524.artifact_schema_errors(changed))

    invalid = dict(success)
    invalid["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="checksum"):
        exp4524.write_artifact(invalid, root=tmp_path)

    monkeypatch.setattr(exp4524, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        exp4524.run(
            root=tmp_path,
            preconditions_checked=_preconditions(),
            measure_lever=lambda lever: _measurement(lever.name, efficiency=0.0),
            offline_reproduction_runner=lambda measurement: {},
            now=lambda: 1.0,
        )
