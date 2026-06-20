"""Tests for Exp 4491 ARC world-model trust energy.

Spec refs: REQ-ARC-WMTE-4491, REQ-ARC-WMTE-4492,
REQ-ARC-WMTE-4493, REQ-ARC-WMTE-4494, SCENARIO-ARC-WMTE-4491,
SCENARIO-ARC-WMTE-4492, SCENARIO-ARC-WMTE-4493.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic.arc_executable_world_model import Transition


def _inc_transitions(n: int = 6) -> list[Transition]:
    transitions: list[Transition] = []
    for i in range(n):
        grid = np.array([[i]], dtype=np.int16)
        transitions.append(
            Transition(
                grid=grid,
                action=1,
                data=None,
                next_grid=grid + 1,
                level_before=0,
                level_after=0,
            )
        )
    return transitions


def _overfits_prefix(grid, action, data):
    value = int(np.asarray(grid)[0, 0])
    return np.asarray(grid) + 1 if value < 4 else np.asarray(grid)


def _generalizes(grid, action, data):
    return np.asarray(grid) + 1


def _noop(grid, action, data):
    return np.asarray(grid)


def test_openspec_declares_exp4491_world_model_trust_energy_contract() -> None:
    """REQ-ARC-WMTE-4491: OpenSpec declares the trust-energy ranking contract."""

    spec = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md").read_text(
        encoding="utf-8"
    )

    for ref in (
        "REQ-ARC-WMTE-4491",
        "REQ-ARC-WMTE-4492",
        "REQ-ARC-WMTE-4493",
        "REQ-ARC-WMTE-4494",
        "SCENARIO-ARC-WMTE-4491",
        "SCENARIO-ARC-WMTE-4492",
        "SCENARIO-ARC-WMTE-4493",
    ):
        assert ref in spec


def test_scenario_arc_wmte_4491_selector_prefers_heldout_generalization() -> None:
    """SCENARIO-ARC-WMTE-4491: trust energy beats first-clears-0.5 baseline."""

    from carnot.agentic.arc_world_model_trust_energy import (
        WorldModelCandidate,
        select_trusted_world_model,
    )

    result = select_trusted_world_model(
        _inc_transitions(),
        [
            WorldModelCandidate("first_prefix_overfit", _overfits_prefix),
            WorldModelCandidate("heldout_generalizer", _generalizes),
            WorldModelCandidate("noop_null", _noop),
        ],
        hidden_state=True,
    )

    assert result.selected.name == "heldout_generalizer"
    assert result.baseline_candidate_name == "first_prefix_overfit"
    assert result.trust_energy_beats_baseline is True
    rows = {row.candidate.name: row for row in result.rows}
    assert rows["first_prefix_overfit"].prefix_accuracy >= 0.5
    assert rows["first_prefix_overfit"].heldout_accuracy == 0.0
    assert rows["heldout_generalizer"].heldout_accuracy == 1.0


def test_req_arc_wmte_4491_selector_rejects_empty_pool_and_scores_single_transition() -> None:
    """REQ-ARC-WMTE-4491: selector has bounded empty and one-row candidate behavior."""

    from carnot.agentic.arc_world_model_trust_energy import (
        WorldModelCandidate,
        select_trusted_world_model,
    )

    with pytest.raises(ValueError, match="at least one world-model candidate"):
        select_trusted_world_model(_inc_transitions(1), (), hidden_state=True)

    result = select_trusted_world_model(
        _inc_transitions(1),
        [WorldModelCandidate("one_row_generalizer", _generalizes)],
        hidden_state=True,
    )

    assert result.selected.name == "one_row_generalizer"
    assert result.selected_score.prefix_accuracy == 1.0
    assert result.selected_score.heldout_accuracy == 1.0


def test_scenario_arc_wmte_4493_null_guard_is_complete_not_fabricated() -> None:
    """SCENARIO-ARC-WMTE-4493: hidden-state null remains terminal and oracle-distinct."""

    from carnot.agentic.arc_world_model_trust_energy import (
        Scorecard,
        build_experiment_artifact,
    )

    artifact = build_experiment_artifact(
        hidden_scorecards=[
            Scorecard(
                game="ka59",
                selected_candidate_name="same_as_baseline",
                baseline_candidate_name="same_as_baseline",
                trust_energy_pick_best=False,
                baseline_pick_best=False,
                n_candidates=2,
                verifier_is_oracle=False,
            )
        ],
        positive_control=Scorecard(
            game="markov_control",
            selected_candidate_name="generalizer",
            baseline_candidate_name="overfit",
            trust_energy_pick_best=True,
            baseline_pick_best=False,
            n_candidates=3,
            verifier_is_oracle=True,
        ),
        preconditions_checked={
            "offline_arcade_import_smoke": True,
            "torch_import": True,
            "torch_version": "test",
        },
        duration_s=1.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["trust_energy_pick_rate"] == 0.0
    assert artifact["baseline_pick_rate"] == 0.0
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_guard"] == "positive_control_passed_hidden_state_null"


def test_req_arc_wmte_4493_positive_control_failure_is_uninformative() -> None:
    """REQ-ARC-WMTE-4493: a failed positive control blocks hidden null interpretation."""

    from carnot.agentic.arc_world_model_trust_energy import (
        Scorecard,
        build_experiment_artifact,
    )

    artifact = build_experiment_artifact(
        hidden_scorecards=[
            Scorecard("ka59", "baseline", "baseline", False, False, 2, False)
        ],
        positive_control=Scorecard(
            "markov_control",
            "baseline",
            "baseline",
            False,
            False,
            2,
            True,
        ),
        preconditions_checked={"offline_arcade_import_smoke": True, "torch_import": True},
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: world_model_trust_energy_positive_control_failed"
    assert artifact["positive_control_passed"] is False
    assert artifact["false_negative_risk_guard"] == "positive_control_failed_null_uninformative"


def test_scenario_arc_wmte_4492_runner_writes_principled_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4492: Exp 4491 writes stable oracle-distinct JSON."""

    from carnot.agentic.arc_world_model_trust_energy import run_experiment_4491

    output_path = tmp_path / "experiment_4491_world_model_trust_energy.json"
    artifact = run_experiment_4491(
        output_path=output_path,
        preconditions_checked={
            "offline_arcade_import_smoke": True,
            "torch_import": True,
            "torch_version": "test",
        },
    )
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert loaded["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert loaded["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert loaded["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert loaded["preconditions_checked"]["torch_import"] is True
    assert loaded["verifier_is_oracle"] is False
    assert loaded["hidden_state_games_n"] >= 11
    assert loaded["trust_energy_pick_rate"] > loaded["baseline_pick_rate"]
    for field in ("honest_verdict", "inference_substrate", "preconditions_checked"):
        assert field in loaded["field_principles"]


def test_req_arc_wmte_4494_live_policy_uses_trust_energy_candidate(monkeypatch) -> None:
    """REQ-ARC-WMTE-4494: live hidden-state planning uses the selector, not a binary gate."""

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    class _FakeProposer:
        def induce(self, game, transitions, cell):
            return True, "ok"

        def world_model_candidates(self, game):
            return [
                ("first_prefix_overfit", _overfits_prefix, None),
                ("heldout_generalizer", _generalizes, None),
            ]

    planned_with: list[str] = []

    def _plan_in_model(engine, is_done, root_grid):
        planned_with.append(engine.__name__)
        return [{"action": 1, "data": None}]

    monkeypatch.setattr(e3, "load_engine", lambda game: (_overfits_prefix, None))
    monkeypatch.setattr(e3, "plan_in_model", _plan_in_model)

    policy = E3AgentPolicy("ka59", proposer=_FakeProposer(), value_head=lambda _frame: 0.0)
    policy.transitions = _inc_transitions()
    policy.root_grid = np.array([[0]], dtype=np.int16)

    policy._induce_and_plan()

    assert planned_with == ["_generalizes"]
    assert policy.plan == [{"action": 1, "data": None}]
    assert policy.world_model_trust_selection.selected.name == "heldout_generalizer"
