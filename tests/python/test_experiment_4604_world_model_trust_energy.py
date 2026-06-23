"""Tests for Exp 4604 oracle-distinct world-model trust energy.

Spec refs: REQ-ARC-WMTE-4604,
SCENARIO-ARC-WMTE-4604-IDENTITY-REJECTED,
SCENARIO-ARC-WMTE-4604-BINARY-CONTROL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic.arc_executable_world_model import Transition


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _noop_transition(i: int) -> Transition:
    grid = np.array([[i, 0], [0, 0]], dtype=np.int16)
    return Transition(grid, 1, None, grid.copy(), 0, 0)


def _two_cell_change_transition(i: int) -> Transition:
    grid = np.array([[i, 0], [0, 0]], dtype=np.int16)
    next_grid = grid.copy()
    next_grid[0, 0] = i + 1
    next_grid[0, 1] = 1
    return Transition(grid, 1, None, next_grid, 0, 0)


def _identity(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    return np.asarray(grid).copy()


def _partial_change_model(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    pred = np.asarray(grid).copy()
    pred[0, 0] = int(pred[0, 0]) + 1
    return pred


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "world_model_verifier_import": True,
        "spec_has_req_4604": True,
        "ok": True,
    }


def test_req_arc_wmte_4604_spec_declares_gate_and_artifact_contract() -> None:
    """REQ-ARC-WMTE-4604: OpenSpec anchors the trust-energy gate contract."""

    from carnot import experiment_4604_world_model_trust_energy as exp4604

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4604" in spec
    assert "SCENARIO-ARC-WMTE-4604-IDENTITY-REJECTED" in spec
    assert "SCENARIO-ARC-WMTE-4604-BINARY-CONTROL" in spec
    assert exp4604.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4604.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4604_identity_engine_is_rejected_on_noop_heavy_games() -> None:
    """SCENARIO-ARC-WMTE-4604-IDENTITY-REJECTED: no-op exact matches do not pass."""

    from carnot.agentic.arc_world_model_trust_energy import (
        binary_exact_gate_pass,
        score_change_weighted_consistency,
    )

    transitions = [_noop_transition(i) for i in range(5)] + [_two_cell_change_transition(5)]

    score = score_change_weighted_consistency(transitions, _identity)

    assert binary_exact_gate_pass(transitions, _identity, threshold=0.5) is True
    assert score.exact_accuracy > 0.5
    assert score.n_changing_transitions == 1
    assert score.correct_changed_cells == 0
    assert score.nondegenerate is False
    assert score.trust_pass is False


def test_req_arc_wmte_4604_change_weighted_score_handles_bad_engines() -> None:
    """REQ-ARC-WMTE-4604: bad engines fail without earning changed-cell credit."""

    from carnot.agentic.arc_world_model_trust_energy import score_change_weighted_consistency

    def _crashes(_grid, _action, _data):
        raise RuntimeError("bad engine")

    def _wrong_shape(_grid, _action, _data):
        return np.zeros((1, 1), dtype=np.int16)

    transitions = [_two_cell_change_transition(0)]

    crash_score = score_change_weighted_consistency(transitions, _crashes)
    shape_score = score_change_weighted_consistency(transitions, _wrong_shape)

    assert crash_score.trust_pass is False
    assert crash_score.correct_changed_cells == 0
    assert shape_score.trust_pass is False
    assert shape_score.correct_changed_cells == 0


def test_req_arc_wmte_4604_selector_passes_imperfect_change_model_binary_rejects() -> None:
    """REQ-ARC-WMTE-4604: changed-cell held-out evidence can pass when exact-match rejects."""

    from carnot.agentic.arc_world_model_trust_energy import (
        WorldModelCandidate,
        select_trusted_world_model,
    )

    result = select_trusted_world_model(
        [_two_cell_change_transition(i) for i in range(6)],
        [
            WorldModelCandidate("identity", _identity),
            WorldModelCandidate("partial_change", _partial_change_model),
        ],
        hidden_state=True,
    )

    rows = {row.candidate.name: row for row in result.rows}
    assert result.selected.name == "partial_change"
    assert rows["partial_change"].heldout_accuracy == 0.0
    assert rows["partial_change"].heldout_change_consistency == 0.5
    assert rows["partial_change"].correct_changed_cells > 0
    assert rows["partial_change"].trust_pass is True
    assert rows["partial_change"].binary_gate_pass is False
    assert rows["identity"].trust_pass is False


def test_scenario_arc_wmte_4604_runner_writes_binary_control_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4604-BINARY-CONTROL: artifact reports the matched control."""

    from carnot import experiment_4604_world_model_trust_energy as exp4604

    artifact = exp4604.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        now=lambda: 1.0,
    )
    loaded = json.loads((tmp_path / exp4604.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["world_model_trust_pass_rate_new"] > artifact["world_model_trust_pass_rate_binary"]
    assert artifact["trust_pass_rate_delta"] > 0.0
    assert artifact["first_win_rate_new"] > 0.0
    assert artifact["first_win_delta"] > 0.0
    assert artifact["first_win_ci"]["ci95"][0] > 0.0
    assert artifact["identity_engine_rejected"] is True
    assert artifact["binary_gate_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["solve_rate_preserved"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["chosen_submitted_config"] == "enable_world_model_trust_energy_gate"
    assert artifact["reproducibility_checksum"] == exp4604.payload_checksum(artifact)
    assert exp4604.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4604_artifact_null_and_error_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4604: null and blocked artifacts remain auditable."""

    from carnot import experiment_4604_world_model_trust_energy as exp4604

    no_plan = exp4604._fixture_plan(exp4604._identity_engine, np.array([[0, 0], [0, 0]]))
    assert no_plan == []

    null_artifact = exp4604.build_artifact(
        preconditions_checked=_preconditions(),
        measurements=[
            {
                "game": "ar25",
                "new_planner_used": False,
                "binary_planner_used": False,
                "new_first_win": False,
                "binary_first_win": False,
                "new_actions_to_first_levelup": None,
                "offline_reproduced": False,
            }
        ],
        identity_control={"identity_engine_rejected": True},
        duration_s=0.0,
    )

    assert null_artifact["honest_verdict"].startswith("complete:")
    assert null_artifact["chosen_submitted_config"] == "unchanged"
    assert "null_delta_methodology_note" in null_artifact
    assert exp4604.artifact_schema_errors(null_artifact) == []

    broken = dict(null_artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["verifier_is_oracle"] = True
    broken["reproducibility_checksum"] = "sha256:bad"
    broken.pop("null_delta_methodology_note")
    errors = exp4604.artifact_schema_errors(broken)
    assert "honest_verdict_terminal_prefix" in errors
    assert "verifier_is_oracle_false" in errors
    assert "reproducibility_checksum" in errors
    assert "null_delta_methodology_note" in errors

    blocked = exp4604.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        now=lambda: 1.0,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert blocked["chosen_submitted_config"] == "unchanged"
