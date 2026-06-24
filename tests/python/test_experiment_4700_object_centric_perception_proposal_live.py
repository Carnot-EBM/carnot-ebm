"""Tests for Exp 4700 object-centric perception proposal conditioning.

Spec refs: REQ-ARC-WMTE-4700,
SCENARIO-ARC-WMTE-4700-PROPOSAL-DIAGNOSTIC,
SCENARIO-ARC-WMTE-4700-LIVE-WIRING.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"

if "coverage" in sys.modules or os.environ.get("CARNOT_SKIP_LIVE_IMPORT_UNDER_COVERAGE") == "1":
    comp = None
else:
    from carnot.agentic import arc_competition_agent as comp


def _frame(values: list[list[int]], *, level: int = 0) -> SimpleNamespace:
    return SimpleNamespace(frame=np.asarray(values, dtype=np.int16), levels_completed=level)


def test_req_arc_wmte_4700_spec_declares_object_centric_contract() -> None:
    """REQ-ARC-WMTE-4700: OpenSpec declares the perception diagnostic artifact."""

    from carnot import experiment_4700_object_centric_perception_proposal_live as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4700" in spec
    assert "SCENARIO-ARC-WMTE-4700-PROPOSAL-DIAGNOSTIC" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4700_object_slots_include_relational_gap_keypoints() -> None:
    """REQ-ARC-WMTE-4700: object slots include deployable constellation gaps."""

    from carnot.agentic.arc_value_learner import (
        ObjectCentricProposalConfig,
        ObjectCentricProposalPolicy,
        object_centric_slots,
    )

    frame = _frame(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 9, 0, 9, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    slots = object_centric_slots(frame, neighborhood_radius=2, max_slots=64)
    assert any(slot["x"] == 1 and slot["y"] == 2 for slot in slots)
    assert any(slot["slot_type"] == "object_neighborhood_gap" for slot in slots)

    policy = ObjectCentricProposalPolicy(
        ObjectCentricProposalConfig(enabled=True, neighborhood_radius=2, max_augmented_clicks=64)
    )
    ranked = policy.rank_candidates(
        frame,
        [
            {"action": 6, "data": {"x": 0, "y": 0}},
            {"action": 6, "data": {"x": 1, "y": 2}},
        ],
    )

    by_point = {
        ((row.get("data") or {}).get("x"), (row.get("data") or {}).get("y")): row
        for row in ranked
        if row["action"] == 6
    }
    assert by_point[(1, 2)]["object_centric_proposal_score"] > by_point[(0, 0)][
        "object_centric_proposal_score"
    ]
    assert by_point[(1, 2)]["object_centric_slot"]["slot_type"] == "object_neighborhood_gap"


def test_scenario_arc_wmte_4700_policy_augments_candidates_before_ranking() -> None:
    """SCENARIO-ARC-WMTE-4700-LIVE-WIRING: deployable slots can add missing clicks."""

    from carnot.agentic.arc_value_learner import (
        ObjectCentricProposalConfig,
        ObjectCentricProposalPolicy,
    )

    frame = _frame(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 8, 0, 8, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    policy = ObjectCentricProposalPolicy(
        ObjectCentricProposalConfig(enabled=True, neighborhood_radius=1, max_augmented_clicks=32)
    )

    ranked = policy.rank_candidates(frame, [{"action": 1, "data": None}])
    click_keys = {
        (row["action"], (row.get("data") or {}).get("x"), (row.get("data") or {}).get("y"))
        for row in ranked
    }

    assert (6, 1, 1) in click_keys
    assert ranked[0]["action"] == 6
    assert policy.diagnostics()["augmented_candidates"] > 0
    assert policy.diagnostics()["verifier_is_oracle"] is False


def test_scenario_arc_wmte_4700_stepwise_orders_object_centric_proposals() -> None:
    """SCENARIO-ARC-WMTE-4700-LIVE-WIRING: StepwiseExplorer uses the policy hook."""

    if comp is None:
        pytest.skip("arc_competition_agent imports the absl/JAX stack under coverage")

    from carnot.agentic.arc_value_learner import ObjectCentricProposalConfig

    explorer = comp.StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        object_centric_proposal=ObjectCentricProposalConfig(
            enabled=True,
            neighborhood_radius=1,
            max_augmented_clicks=16,
        ),
    )
    frame = _frame([[0, 0, 0], [0, 7, 0], [0, 0, 0]])

    ranked = explorer._apply_object_centric_proposal_order(
        frame,
        [{"action": 1, "data": None}],
        previous_frame=None,
    )

    assert ranked[0]["action"] == 6
    assert explorer.object_centric_proposal_diagnostics()["enabled"] is True


def test_scenario_arc_wmte_4700_artifact_records_diagnostic_null() -> None:
    """SCENARIO-ARC-WMTE-4700-PROPOSAL-DIAGNOSTIC: coverage drives the residual."""

    from carnot import experiment_4700_object_centric_perception_proposal_live as mod

    coverage = {
        "order1": {"coverage": 0.0, "covered_steps": 0, "total_steps": 4},
        "object_centric": {"coverage": 0.25, "covered_steps": 1, "total_steps": 4},
        "upper_bound_ceiling": {
            "coverage": 1.0,
            "covered_steps": 4,
            "total_steps": 4,
            "deployable": False,
        },
    }
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        target_game="r11l",
        proposal_coverage_by_representation=coverage,
        object_result={"reached_level": 0, "offline_reproduced": False, "reproduced_levels": 0},
        order1_result={"reached_level": 0, "offline_reproduced": False, "reproduced_levels": 0},
        bare_control_passed=True,
        offpath_calibrated=True,
        duration_s=60.0,
    )

    assert artifact["perception_is_the_wall"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["residual_cause_hypothesis"] == "offpath_calibration_insufficient"
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4700_submitted_default_records_object_policy_off() -> None:
    """REQ-ARC-WMTE-4700: submitted default is explicit until a reproduced win exists."""

    if comp is None:
        pytest.skip("arc_competition_agent imports the absl/JAX stack under coverage")

    policy = comp.E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)

    assert comp.SUBMITTED_AGENT_CONFIG["object_centric_proposal_enabled"] is False
    assert policy.explorer.object_centric_proposal_diagnostics()["enabled"] is False
