"""Tests for Exp 4791 S2 off-path structural-energy trust gate.

Spec refs: REQ-ARC-WMTE-4791,
SCENARIO-ARC-WMTE-4791-OFFPATH-TRUST-GATE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4791_structural_energy_s2_offpath_trust_gate as mod
from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier
from carnot.agentic.arc_world_model_trust_energy import (
    WorldModelCandidate,
    select_trusted_world_model,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4791")
    end = spec.index("### REQ-ARC-WMTE-4768", start)
    return spec[start:end]


def _transition_rows() -> list[Transition]:
    rows: list[Transition] = []
    for i in range(6):
        grid = np.zeros((3, 3), dtype=np.int16)
        grid[1, 1] = i
        next_grid = np.array(grid, copy=True)
        next_grid[1, 1] = i + 1
        rows.append(Transition(grid, 1, None, next_grid, 0, 0))
    return rows


def _prefix_overfit(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    value = int(np.asarray(grid)[1, 1])
    out = np.array(grid, copy=True)
    if value < 4:
        out[1, 1] += 1
    return out


def _generalizer(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[1, 1] += 1
    return out


def _noop(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    return np.asarray(grid)


class _CenterChangeScorer:
    energy_config = {"source": "test_center_change_scorer"}

    def transition_energy(
        self,
        previous_grid: np.ndarray,
        _action: int,
        _data: Any,
        predicted_grid: np.ndarray,
    ) -> float:
        prev = np.asarray(previous_grid)
        pred = np.asarray(predicted_grid)
        if pred.shape != prev.shape:
            return 999.0
        center_changed = pred[1, 1] != prev[1, 1]
        collateral_changed = int(np.count_nonzero(pred != prev)) - int(center_changed)
        return float(collateral_changed - int(center_changed))


def test_req_arc_wmte_4791_spec_declares_s2_contract() -> None:
    """REQ-ARC-WMTE-4791: OpenSpec declares the S2 trust-gate contract first."""

    section = _normalise(_spec_section())

    assert "REQ-ARC-WMTE-4791" in section
    assert "SCENARIO-ARC-WMTE-4791-OFFPATH-TRUST-GATE" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "WorldModelVerifier" in section
    assert "E3AgentPolicy" in section
    assert "off-path structural energy" in section
    assert "binary `WorldModelVerifier` exact-accuracy" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert field in section
        assert _normalise(principle["principle"]) in section


def test_scenario_arc_wmte_4791_world_model_verifier_ranks_offpath_energy() -> None:
    """SCENARIO-ARC-WMTE-4791-OFFPATH-TRUST-GATE: verifier ranks predictions without oracle labels."""

    verifier = WorldModelVerifier(_transition_rows()[-2:])
    scores = verifier.rank_offpath_structural_energy(
        [
            ("prefix_overfit", _prefix_overfit),
            ("generalizer", _generalizer),
            ("noop", _noop),
        ],
        energy_scorer=_CenterChangeScorer(),
    )

    assert [row["candidate_name"] for row in scores] == ["generalizer", "noop", "prefix_overfit"]
    assert scores[0]["offpath_structural_energy"] < scores[1]["offpath_structural_energy"]
    assert all("cell_recall" not in row for row in scores)


def test_scenario_arc_wmte_4791_live_selector_uses_energy_not_binary_gate() -> None:
    """REQ-ARC-WMTE-4791: live selector uses E-ranking while retaining the binary control."""

    selection = select_trusted_world_model(
        _transition_rows(),
        [
            WorldModelCandidate("prefix_overfit", _prefix_overfit),
            WorldModelCandidate("generalizer", _generalizer),
            WorldModelCandidate("noop", _noop),
        ],
        hidden_state=True,
        offpath_energy_scorer=_CenterChangeScorer(),
    )

    assert selection.selected.name == "generalizer"
    assert selection.baseline_candidate_name == "prefix_overfit"
    assert selection.verifier_is_oracle is False
    rows = {row.candidate.name: row for row in selection.rows}
    assert rows["generalizer"].trust_energy < rows["prefix_overfit"].trust_energy
    assert rows["prefix_overfit"].binary_gate_pass is True


def test_scenario_arc_wmte_4791_builds_success_artifact_with_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4791-OFFPATH-TRUST-GATE: artifact reports the load-bearing delta."""

    candidates = [
        WorldModelCandidate("prefix_overfit", _prefix_overfit),
        WorldModelCandidate("generalizer", _generalizer),
        WorldModelCandidate("noop", _noop),
    ]
    game_results = [
        mod.evaluate_candidate_set(
            game=f"hidden_{i}",
            transitions=_transition_rows(),
            candidates=candidates,
            energy_scorer=_CenterChangeScorer(),
        )
        for i in range(mod.MIN_HELDOUT_GAMES)
    ]
    artifact = mod.build_artifact(
        game_results,
        preconditions_checked={
            "offline_arcade": True,
            "world_model_verifier_import": True,
            "s1_artifact_read": True,
        },
        live_path_reachable=True,
        random_seed=4791,
        bootstrap_resamples=200,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "success_structural_energy_s2_trust_gate_authorizes_s3"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["binary_accuracy_control"]["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["energy_selected_offpath_cell_recall"] == pytest.approx(1.0)
    assert artifact["accuracy_gate_selected_offpath_cell_recall"] == pytest.approx(0.0)
    assert artifact["energy_minus_accuracy_delta_ci95"][0] > 0.0
    assert artifact["n_heldout_games"] == mod.MIN_HELDOUT_GAMES
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["s3_authorized"] is True

    out = mod.write_artifact(artifact, root=tmp_path)
    assert out == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4791_bounded_and_blocked_paths() -> None:
    """REQ-ARC-WMTE-4791: failed deltas and failed preconditions fail closed."""

    tied = [
        mod.GameTrustGateResult(
            game=f"tied_{i}",
            n_candidates=2,
            energy_selected_candidate="a",
            accuracy_gate_selected_candidate="a",
            energy_selected_offpath_cell_recall=0.5,
            accuracy_gate_selected_offpath_cell_recall=0.5,
            energy_minus_accuracy_delta=0.0,
            energy_selected_structural_energy=0.0,
            accuracy_gate_exact_accuracy=1.0,
            accuracy_gate_passed=True,
            candidate_rows=[],
        )
        for i in range(mod.MIN_HELDOUT_GAMES)
    ]
    bounded = mod.build_artifact(
        tied,
        preconditions_checked={"offline_arcade": True, "world_model_verifier_import": True},
        live_path_reachable=True,
        random_seed=4791,
        bootstrap_resamples=50,
    )
    mod.validate_artifact(bounded)
    assert bounded["honest_verdict"] == "complete_structural_energy_s2_no_live_trust_value"
    assert bounded["s3_authorized"] is False

    blocked = mod.build_blocked_artifact(
        "blocked_world_model_verifier_missing",
        {"offline_arcade": True, "world_model_verifier_import": False},
        random_seed=4791,
    )
    mod.validate_artifact(blocked)
    assert blocked["energy_selected_offpath_cell_recall"] is None
    assert blocked["accuracy_gate_selected_offpath_cell_recall"] is None
    assert blocked["n_heldout_games"] == 0

    bad_artifact = copy.deepcopy(bounded)
    bad_artifact["verifier_is_oracle"] = True
    bad_artifact["reproducibility_checksum"] = mod._checksum_payload(bad_artifact)
    with pytest.raises(ValueError):
        mod.validate_artifact(bad_artifact)


def test_req_arc_wmte_4791_preconditions_and_live_path_smokes() -> None:
    """REQ-ARC-WMTE-4791: precondition and lint helpers are deterministic."""

    checks = mod.check_preconditions()
    assert checks["offline_arcade"] is True
    assert checks["world_model_verifier_import"] is True

    assert mod._bootstrap_mean_ci([], seed=1) is None
    assert mod._bootstrap_mean_ci([0.25], seed=1) == [0.25, 0.25]
    assert mod._ci_excludes_zero_positive([0.01, 0.02]) is True
    assert mod._ci_excludes_zero_positive([0.0, 0.02]) is False
    assert mod._ci_excludes_zero_positive(None) is False
