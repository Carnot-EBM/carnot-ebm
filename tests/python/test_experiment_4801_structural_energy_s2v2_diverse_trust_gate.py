"""Tests for Exp 4801 S2-v2 diverse structural-energy trust gate.

Spec refs: REQ-ARC-WMTE-4801,
SCENARIO-ARC-WMTE-4801-DIVERSE-TRUST-GATE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4801_structural_energy_s2v2_diverse_trust_gate as mod
from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_world_model_trust_energy import WorldModelCandidate
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4801")
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


def _collateral(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    out = _generalizer(grid, _action, _data)
    out[0, 0] = 9
    return out


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


def _manual_result(
    game: str,
    *,
    energy_recall: float,
    accuracy_recall: float,
    oracle_recall: float,
) -> mod.GameTrustGateResult:
    candidate_rows = [
        {
            "candidate_name": f"{game}/accuracy_seed",
            "candidate_source": "cached_induced_engine",
            "genuinely_induced": True,
            "prefix_accuracy": 0.55,
            "heldout_accuracy": 0.31,
            "heldout_cell_recall": accuracy_recall,
            "offpath_structural_energy": 2.0,
            "binary_gate_pass": True,
        },
        {
            "candidate_name": f"{game}/energy_seed",
            "candidate_source": "programmatic_expert_induction",
            "genuinely_induced": True,
            "prefix_accuracy": 0.42,
            "heldout_accuracy": 0.28,
            "heldout_cell_recall": energy_recall,
            "offpath_structural_energy": 1.0,
            "binary_gate_pass": False,
        },
        {
            "candidate_name": f"{game}/headroom_seed",
            "candidate_source": "programmatic_expert_induction",
            "genuinely_induced": True,
            "prefix_accuracy": 0.25,
            "heldout_accuracy": 0.23,
            "heldout_cell_recall": oracle_recall,
            "offpath_structural_energy": 3.0,
            "binary_gate_pass": False,
        },
    ]
    return mod.GameTrustGateResult.from_candidate_rows(
        game=game,
        energy_selected_candidate=f"{game}/energy_seed",
        accuracy_gate_selected_candidate=f"{game}/accuracy_seed",
        candidate_rows=candidate_rows,
    )


def test_req_arc_wmte_4801_spec_declares_s2v2_contract() -> None:
    """REQ-ARC-WMTE-4801: OpenSpec declares the S2-v2 contract first."""

    section = _normalise(_spec_section())

    assert "REQ-ARC-WMTE-4801" in section
    assert "SCENARIO-ARC-WMTE-4801-DIVERSE-TRUST-GATE" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "n_effective_games" in section
    assert "candidate_rows[*].heldout_cell_recall" in section
    assert "programmatic expert-induction" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert field in section
        assert _normalise(principle["principle"]) in section


def test_scenario_arc_wmte_4801_logs_exact_diverse_selection_rows() -> None:
    """SCENARIO-ARC-WMTE-4801-DIVERSE-TRUST-GATE: row recalls drive selection deltas."""

    candidates = [
        WorldModelCandidate("prefix_overfit", _prefix_overfit),
        WorldModelCandidate("generalizer", _generalizer),
        WorldModelCandidate("collateral", _collateral),
    ]
    metadata = {
        candidate.name: {
            "candidate_source": "programmatic_expert_induction",
            "genuinely_induced": True,
            "generation_round": i,
        }
        for i, candidate in enumerate(candidates)
    }
    result = mod.evaluate_candidate_set(
        game="unit_game",
        transitions=_transition_rows(),
        candidates=candidates,
        candidate_metadata=metadata,
        energy_scorer=_CenterChangeScorer(),
    )
    rows = {row["candidate_name"]: row for row in result.candidate_rows}

    assert result.energy_selected_candidate == "generalizer"
    assert result.accuracy_gate_selected_candidate == "prefix_overfit"
    assert result.selection_candidates_differ is True
    assert result.effective is True
    assert result.heldout_cell_recall_spread == pytest.approx(1.0)
    assert rows["generalizer"]["heldout_cell_recall"] == pytest.approx(1.0)
    assert rows["prefix_overfit"]["heldout_cell_recall"] == pytest.approx(0.0)
    assert result.energy_selected_offpath_cell_recall == rows["generalizer"]["heldout_cell_recall"]
    assert (
        result.accuracy_gate_selected_offpath_cell_recall
        == rows["prefix_overfit"]["heldout_cell_recall"]
    )
    assert all(row["genuinely_induced"] is True for row in result.candidate_rows)


def test_req_arc_wmte_4801_builds_bounded_diverse_artifact_with_positive_control(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4801: bounded null requires a diverse pool and positive control."""

    game_results = [
        _manual_result("g0", energy_recall=0.20, accuracy_recall=0.60, oracle_recall=0.70),
        _manual_result("g1", energy_recall=0.30, accuracy_recall=0.50, oracle_recall=0.65),
        _manual_result("g2", energy_recall=0.65, accuracy_recall=0.55, oracle_recall=0.75),
        _manual_result("g3", energy_recall=0.55, accuracy_recall=0.45, oracle_recall=0.72),
        _manual_result("g4", energy_recall=0.50, accuracy_recall=0.35, oracle_recall=0.80),
    ]
    artifact = mod.build_artifact(
        game_results,
        preconditions_checked={
            "offline_arcade": True,
            "world_model_verifier_import": True,
            "s1_artifact_read": True,
        },
        live_path_reachable=True,
        random_seed=4801,
        bootstrap_resamples=200,
        duration_s=1.25,
        energy_scorer=_CenterChangeScorer(),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete_structural_energy_s2v2_bounded_diverse_pool"
    assert artifact["n_effective_games"] == 5
    assert artifact["min_heldout_games"] == 5
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["candidates_genuinely_induced"] is True
    assert artifact["s3_authorized"] is False
    assert artifact["game_results"] == [row.to_json() for row in game_results]

    flags: list[adversarial_verify.Flag] = []
    adversarial_verify.check_engine_selection_candidate_diversity(artifact, flags)
    assert flags == []

    out = mod.write_artifact(artifact, root=tmp_path)
    assert out == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4801_success_and_inconclusive_paths() -> None:
    """REQ-ARC-WMTE-4801: only five effective diverse games can pass."""

    positive = [
        _manual_result(f"p{i}", energy_recall=0.70, accuracy_recall=0.40, oracle_recall=0.75)
        for i in range(mod.MIN_EFFECTIVE_GAMES)
    ]
    success = mod.build_artifact(
        positive,
        preconditions_checked={"offline_arcade": True, "world_model_verifier_import": True},
        live_path_reachable=True,
        random_seed=4801,
        bootstrap_resamples=50,
        energy_scorer=_CenterChangeScorer(),
    )
    mod.validate_artifact(success)
    assert success["honest_verdict"] == "success_structural_energy_s2v2_trust_gate_authorizes_s3"
    assert success["s3_authorized"] is True

    insufficient = mod.build_artifact(
        positive[:4],
        preconditions_checked={"offline_arcade": True, "world_model_verifier_import": True},
        live_path_reachable=True,
        random_seed=4801,
        bootstrap_resamples=50,
        energy_scorer=_CenterChangeScorer(),
    )
    mod.validate_artifact(insufficient)
    assert (
        insufficient["honest_verdict"]
        == "complete_structural_energy_s2v2_inconclusive_insufficient_candidate_diversity"
    )
    assert insufficient["game_results"] == []
    assert insufficient["n_effective_games"] == 4
    assert insufficient["energy_minus_accuracy_delta"] is None

    blocked = mod.build_blocked_artifact(
        "blocked_world_model_verifier_missing",
        {"offline_arcade": True, "world_model_verifier_import": False},
        random_seed=4801,
    )
    mod.validate_artifact(blocked)
    assert blocked["n_effective_games"] == 0
    assert blocked["energy_selected_offpath_cell_recall"] is None


def test_req_arc_wmte_4801_validation_rejects_fabricated_effective_count() -> None:
    """REQ-ARC-WMTE-4801: n_effective_games must match candidate_rows spread."""

    artifact = mod.build_artifact(
        [
            _manual_result(f"g{i}", energy_recall=0.60, accuracy_recall=0.40, oracle_recall=0.70)
            for i in range(mod.MIN_EFFECTIVE_GAMES)
        ],
        preconditions_checked={"offline_arcade": True, "world_model_verifier_import": True},
        live_path_reachable=True,
        random_seed=4801,
        bootstrap_resamples=50,
        energy_scorer=_CenterChangeScorer(),
    )

    bad = copy.deepcopy(artifact)
    bad["n_effective_games"] = 1
    bad["reproducibility_checksum"] = mod._checksum_payload(bad)
    with pytest.raises(ValueError, match="n_effective_games"):
        mod.validate_artifact(bad)

    bad_pc = copy.deepcopy(artifact)
    bad_pc["honest_verdict"] = "complete_structural_energy_s2v2_bounded_diverse_pool"
    bad_pc["positive_control_passed"] = False
    bad_pc["s3_authorized"] = False
    bad_pc["reproducibility_checksum"] = mod._checksum_payload(bad_pc)
    with pytest.raises(ValueError, match="positive_control_passed"):
        mod.validate_artifact(bad_pc)


def test_req_arc_wmte_4801_edge_guards_cover_degenerate_branches() -> None:
    """REQ-ARC-WMTE-4801: helper guards cover malformed and no-headroom inputs."""

    assert mod._clean_float("not-a-number") is None
    assert mod._bootstrap_mean_ci([0.25], seed=1) == [0.25, 0.25]
    assert mod._row_by_candidate_name([{"candidate_name": "a"}], "missing") == {
        "candidate_name": "a"
    }
    assert mod._candidate_by_name([WorldModelCandidate("a", _generalizer)], None).name == "a"
    with pytest.raises(ValueError, match="at least one candidate"):
        mod.evaluate_candidate_set(game="empty", transitions=_transition_rows(), candidates=[])

    same_selector = mod.GameTrustGateResult.from_candidate_rows(
        game="same",
        energy_selected_candidate="same/accuracy_seed",
        accuracy_gate_selected_candidate="same/accuracy_seed",
        candidate_rows=[
            {
                "candidate_name": "same/accuracy_seed",
                "genuinely_induced": True,
                "heldout_cell_recall": 0.6,
                "offpath_structural_energy": 1.0,
                "binary_gate_pass": True,
            },
            {
                "candidate_name": "same/other_seed",
                "genuinely_induced": True,
                "heldout_cell_recall": 0.4,
                "offpath_structural_energy": 2.0,
                "binary_gate_pass": False,
            },
            {
                "candidate_name": "same/third_seed",
                "genuinely_induced": True,
                "heldout_cell_recall": 0.5,
                "offpath_structural_energy": 3.0,
                "binary_gate_pass": False,
            },
        ],
    )
    negative_different = _manual_result(
        "negative",
        energy_recall=0.20,
        accuracy_recall=0.60,
        oracle_recall=0.70,
    )
    broken_positive = _manual_result(
        "broken",
        energy_recall=0.20,
        accuracy_recall=0.0,
        oracle_recall=0.30,
    )
    assert mod._positive_control_passed([same_selector]) is False
    assert mod._success_not_sabotage([same_selector]) is True
    assert mod._success_not_sabotage([negative_different]) is True
    assert mod._success_not_sabotage([broken_positive]) is False
    assert (
        mod._linter_effective_game_count(
            [
                "bad",
                {"candidate_rows": []},
                {"candidate_rows": [{"heldout_cell_recall": 0.1}, {"heldout_cell_recall": 0.4}]},
            ]
        )
        == 1
    )

    no_headroom = mod.build_artifact(
        [
            _manual_result(f"nh{i}", energy_recall=0.50, accuracy_recall=0.70, oracle_recall=0.60)
            for i in range(mod.MIN_EFFECTIVE_GAMES)
        ],
        preconditions_checked={"offline_arcade": True, "world_model_verifier_import": True},
        live_path_reachable=True,
        random_seed=4801,
        bootstrap_resamples=50,
        energy_scorer=_CenterChangeScorer(),
    )
    mod.validate_artifact(no_headroom)
    assert no_headroom["honest_verdict"] == (
        "complete_structural_energy_s2v2_inconclusive_insufficient_candidate_diversity"
    )
