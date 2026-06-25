"""Tests for Exp 4713 present-winner surfacing ranker.

Spec refs: REQ-ARC-WMTE-4713,
SCENARIO-ARC-WMTE-4713-PRECISION-AT-K,
SCENARIO-ARC-WMTE-4713-LIVE-ABLATION.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys

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


def test_req_arc_wmte_4713_spec_declares_surface_present_winner_contract() -> None:
    """REQ-ARC-WMTE-4713: OpenSpec anchors the surfacing artifact."""

    from carnot import experiment_4713_surface_present_winner_verifier_ranker as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4713" in spec
    assert "SCENARIO-ARC-WMTE-4713-PRECISION-AT-K" in spec
    assert "SCENARIO-ARC-WMTE-4713-LIVE-ABLATION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4713_ranker_lifts_calibrated_candidate() -> None:
    """SCENARIO-ARC-WMTE-4713-PRECISION-AT-K: calibrated features rerank the pool."""

    from carnot.agentic.arc_value_learner import (
        ObjectCentricProposalConfig,
        ObjectCentricProposalPolicy,
        OffPathCalibratedProposalRanker,
    )

    ranker = OffPathCalibratedProposalRanker()
    ranker.fit(
        [
            {"features": [1.0, 0.0], "label": 1},
            {"features": [0.9, 0.1], "label": 1},
            {"features": [0.0, 1.0], "label": 0},
            {"features": [0.1, 0.9], "label": 0},
        ]
    )
    rows = ranker.rank_rows(
        [
            {"action": 1, "data": None, "surfacing_features": [0.0, 1.0]},
            {"action": 2, "data": None, "surfacing_features": [1.0, 0.0]},
        ]
    )

    assert rows[0]["action"] == 2
    assert rows[0]["surfacing_verifier_score"] > rows[1]["surfacing_verifier_score"]
    assert ranker.diagnostics()["offpath_calibrated"] is True
    assert ranker.diagnostics()["verifier_is_oracle"] is False

    policy = ObjectCentricProposalPolicy(
        ObjectCentricProposalConfig(enabled=True, surfacing_ranker_enabled=True)
    )
    policy.calibrate_surfacing_ranker(
        [
            {"features": [1.0, 0.0], "label": 1},
            {"features": [0.9, 0.1], "label": 1},
            {"features": [0.0, 1.0], "label": 0},
            {"features": [0.1, 0.9], "label": 0},
        ]
    )
    ranked = policy.rank_candidates(
        _frame([[0, 0], [0, 0]]),
        [
            {"action": 1, "data": None, "surfacing_features": [0.0, 1.0]},
            {"action": 2, "data": None, "surfacing_features": [1.0, 0.0]},
        ],
    )

    assert ranked[0]["action"] == 2
    assert ranked[0]["surfacing_verifier_score"] > ranked[1]["surfacing_verifier_score"]
    assert policy.diagnostics()["surfacing_ranker"]["offpath_calibrated"] is True


def test_scenario_arc_wmte_4713_stepwise_uses_surfacing_ranker_hook() -> None:
    """SCENARIO-ARC-WMTE-4713-LIVE-ABLATION: StepwiseExplorer reaches the ranker hook."""

    if comp is None:
        pytest.skip("arc_competition_agent imports the absl/JAX stack under coverage")

    from carnot.agentic.arc_value_learner import (
        ObjectCentricProposalConfig,
        ObjectCentricProposalPolicy,
    )

    policy = ObjectCentricProposalPolicy(
        ObjectCentricProposalConfig(enabled=True, surfacing_ranker_enabled=True)
    )
    policy.calibrate_surfacing_ranker(
        [
            {"features": [1.0, 0.0], "label": 1},
            {"features": [0.0, 1.0], "label": 0},
        ]
    )
    explorer = comp.StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        object_centric_proposal=policy,
    )

    ranked = explorer._apply_object_centric_proposal_order(
        _frame([[0, 0], [0, 0]]),
        [
            {"action": 1, "data": None, "surfacing_features": [0.0, 1.0]},
            {"action": 2, "data": None, "surfacing_features": [1.0, 0.0]},
        ],
        previous_frame=None,
    )

    assert ranked[0]["action"] == 2
    assert explorer.object_centric_proposal_diagnostics()["surfacing_ranker"][
        "offpath_calibrated"
    ] is True


def test_req_arc_wmte_4713_artifact_schema_records_honest_null() -> None:
    """REQ-ARC-WMTE-4713: artifact fields are checksummed and oracle-distinct."""

    from carnot import experiment_4713_surface_present_winner_verifier_ranker as mod

    artifact = mod.build_artifact(
        preconditions_checked={
            "ok": True,
            "a1_operator_importable": True,
            "qwen3_5_9b_mtp_gguf_cached": True,
            "offline_arcade": True,
            "qwen_proposer_port_verified": True,
        },
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        target_game="r11l",
        winner_present_coverage=1.0,
        winner_rank_pre_surfacing=[59, 161, 12, 77],
        precision_at_k_no_surfacing={"k": 8, "hits": 0, "total": 4, "precision": 0.0},
        precision_at_k_with_surfacing={"k": 8, "hits": 0, "total": 4, "precision": 0.0},
        surfacing_result={"reached_level": 0, "offline_reproduced": False, "reproduced_levels": 0},
        no_surfacing_result={"reached_level": 0},
        offpath_calibrated=True,
        bare_control_passed=True,
        missing_verifier_gap_logged=True,
        residual_cause="present_winner_not_separable_from_distractors",
        duration_s=60.0,
    )

    assert artifact["honest_verdict"] == (
        "complete: surface_present_winner_no_new_level_residual_"
        "present_winner_not_separable_from_distractors"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
