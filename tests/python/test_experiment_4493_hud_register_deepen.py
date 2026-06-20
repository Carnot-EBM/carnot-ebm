"""Tests for Exp 4493 HUD/register state deepening.

Spec refs: REQ-ARC-WMTE-4495, SCENARIO-ARC-WMTE-4494.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_4493_hud_register_deepen as exp4493


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions() -> dict[str, object]:
    return {
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "test",
    }


def test_req_arc_wmte_4495_spec_declares_hud_register_artifact() -> None:
    """REQ-ARC-WMTE-4495: OpenSpec names the register-state artifact and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in ("REQ-ARC-WMTE-4495", "SCENARIO-ARC-WMTE-4494"):
        assert ref in spec
    assert exp4493.RESULT_RELATIVE_PATH in spec
    for phrase in (
        "(grid, registers)",
        "hud_count",
        "undo-stack",
        "goal_predicate_heldout_score",
        "offline_reproduced=true",
        "reproduced_levels >= 1",
    ):
        assert phrase in spec
    for field, principle in exp4493.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4495_registered_state_key_and_ka59_goal_use_hud_count() -> None:
    """REQ-ARC-WMTE-4495: ka59 completion reads hud_count from registered state."""

    grid = np.zeros((4, 4), dtype=np.int16)
    grid[0, 0] = 4

    grid_only = exp4493.RegisteredState(grid=grid, registers={})
    register_done = exp4493.RegisteredState(grid=grid, registers={"hud_count": 0})
    register_blocked = exp4493.RegisteredState(grid=grid, registers={"hud_count": 1})

    assert exp4493.ka59_is_level_complete(grid_only) is False
    assert exp4493.ka59_is_level_complete(register_done) is True
    assert exp4493.ka59_is_level_complete(register_blocked) is False
    assert exp4493.state_key(register_done) == exp4493.state_key((grid, {"hud_count": 0}))
    assert exp4493.state_key(register_done) != exp4493.state_key(register_blocked)

    normalised = exp4493.normalise_registered_state((grid, {"undo_stack_depth": 2}))
    assert np.array_equal(normalised.grid, grid)
    assert normalised.registers == {"undo_stack_depth": 2}

    assert exp4493.induce_registers("ka59", grid) == {"hud_count": 1}
    assert exp4493.induce_registers("ar25", grid, {"undo_stack": [grid.tolist()]}) == {
        "undo_stack_depth": 1
    }
    assert exp4493.induce_registers("other", grid) == {}
    assert exp4493.registered_is_level_complete("ka59", register_done) is True
    assert exp4493.registered_is_level_complete("ar25", np.zeros((3, 3), dtype=np.int16)) is True
    ar25_blocked = np.zeros((3, 3), dtype=np.int16)
    ar25_blocked[0, 0] = 1
    ar25_blocked[0, 1] = 1
    assert exp4493.registered_is_level_complete("ar25", ar25_blocked) is False
    with pytest.raises(ValueError, match="unsupported game"):
        exp4493.registered_is_level_complete("bad", register_done)


def test_scenario_arc_wmte_4494_goal_report_scores_registered_predicate_separately() -> None:
    """SCENARIO-ARC-WMTE-4494: GOAL score is held out from grid-only dynamics."""

    report = exp4493.build_goal_accountability_report()

    assert report["goal_predicate_heldout_score"] == pytest.approx(1.0)
    assert report["grid_only_goal_predicate_heldout_score"] < 1.0
    assert report["goal_examples_n"] == 4
    assert report["register_state_contract"]["state_shape"] == "(grid, registers)"
    assert "hud_count" in report["register_state_contract"]["registers"]
    assert "undo_stack_depth" in report["register_state_contract"]["registers"]

    assert exp4493.score_goal_predicate([], exp4493.ka59_is_level_complete) == {
        "n": 0,
        "correct": 0,
        "accuracy": None,
    }


def test_req_arc_wmte_4495_success_artifact_gate_and_schema_errors(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4495: success requires offline L2 reproduction and principled fields."""

    attempt = exp4493.ReproductionAttempt(
        game="ar25",
        plan_label="ar25_register_probe",
        claimed_level=2,
        reached_level=2,
        reproduced=True,
        residual=None,
    )
    artifact = exp4493.build_artifact(
        preconditions_checked=_preconditions(),
        attempts=[attempt],
        goal_report=exp4493.build_goal_accountability_report(),
        tests_pass=True,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["goal_predicate_heldout_score"] == pytest.approx(1.0)
    assert artifact["schema_errors"] == []
    assert exp4493.artifact_schema_errors(artifact) == []

    out = exp4493.write_artifact(artifact, root=tmp_path)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    mutations = [
        (lambda item: item.pop("honest_verdict"), "missing required"),
        (lambda item: item.__setitem__("honest_verdict", "validated"), "terminal prefix"),
        (
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
            "inference_substrate",
        ),
        (lambda item: item.__setitem__("preconditions_checked", []), "preconditions_checked"),
        (lambda item: item.__setitem__("field_principles", {}), "field_principles"),
        (lambda item: item.__setitem__("candidate_reproduction_attempts", []), "attempt"),
        (lambda item: item.__setitem__("goal_predicate_heldout_score", None), "goal"),
        (lambda item: item.__setitem__("offline_reproduced", False), "success artifact"),
        (lambda item: item.__setitem__("reproduced_levels", 0), "success artifact"),
    ]

    for mutate, expected in mutations:
        changed = dict(artifact)
        mutate(changed)
        assert any(expected in error for error in exp4493.artifact_schema_errors(changed))

    fabricated = dict(artifact)
    fabricated["offline_reproduced"] = False
    with pytest.raises(ValueError, match="success artifact"):
        exp4493.write_artifact(fabricated, root=tmp_path)


def test_scenario_arc_wmte_4494_runner_writes_success_with_injected_l2(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4494: injected offline replay success writes stable JSON."""

    calls: list[tuple[str, int]] = []

    def fake_runner(game: str, labels: tuple[str, ...], apply_fn: object, claimed_level: int):
        calls.append((game, len(labels)))
        if game == "ar25":
            return {"reached_level": 2, "reproduced": True}
        return {"reached_level": 1, "reproduced": False, "residual": "still_l1_only"}

    artifact = exp4493.run_experiment(
        root=tmp_path,
        reproduction_runner=fake_runner,
        preconditions_checked=_preconditions(),
        tests_pass=True,
    )
    written = json.loads((tmp_path / exp4493.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert calls[0][0] == "ar25"
    assert artifact == written
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["candidate_reproduction_attempts"][0]["game"] == "ar25"


def test_scenario_arc_wmte_4494_runner_reports_honest_residual_when_l2_blocks(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4494: no replay success emits terminal residual, not fabrication."""

    def fake_runner(game: str, labels: tuple[str, ...], apply_fn: object, claimed_level: int):
        return {"reached_level": 1, "reproduced": False, "residual": f"{game}_l2_residual"}

    artifact = exp4493.run_experiment(
        root=tmp_path,
        reproduction_runner=fake_runner,
        preconditions_checked=_preconditions(),
        tests_pass=True,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["residual_blockers"] == ["ar25_l2_residual", "ka59_l2_residual"]
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4495_runner_rejects_missing_resource_preconditions() -> None:
    """REQ-ARC-WMTE-4495: missing import or torch preconditions block before replay."""

    with pytest.raises(RuntimeError, match="blocked_offline_arcade_import_smoke"):
        exp4493.ensure_preconditions_ready(
            {"offline_arcade_import_smoke": False, "torch_import": True}
        )
    with pytest.raises(RuntimeError, match="blocked_torch_import"):
        exp4493.ensure_preconditions_ready(
            {"offline_arcade_import_smoke": True, "torch_import": False}
        )
