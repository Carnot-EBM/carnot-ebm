"""Tests for Exp5630 ARC epistemic object-probe prototype artifact.

Spec refs: REQ-ARC-WMTE-5630,
SCENARIO-ARC-WMTE-5630-INFORMATIVE-PROBE-POSITIVE,
SCENARIO-ARC-WMTE-5630-NEGATIVE-AND-UNSAFE-REJECTION.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot import experiment_5630_arc_epistemic_object_probe_prototype as mod
from carnot.agentic.arc_epistemic_object_probe import ObjectProbeObservation


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _grid(*, left_color: int = 5, right_color: int = 7) -> np.ndarray:
    grid = np.zeros((8, 12), dtype=np.int16)
    grid[2:4, 2:4] = left_color
    grid[2:4, 8:10] = right_color
    return grid


def _obs(game: str, step: int, x: int, y: int) -> ObjectProbeObservation:
    before = _grid()
    after = before.copy()
    if x < 6:
        after[2:4, 2:4] = 9
    else:
        after[2:4, 8:10] = 9
    return ObjectProbeObservation(
        trace_id=game,
        step=step,
        state=before,
        action=6,
        data={"x": x, "y": y},
        successor=after,
        level_before=0,
        level_after=0,
    )


def _traces() -> dict[str, list[ObjectProbeObservation]]:
    return {
        "dc22": [_obs("dc22", 0, 3, 3), _obs("dc22", 1, 9, 3)],
        "bp35": [_obs("bp35", 0, 3, 3), _obs("bp35", 1, 9, 3)],
        "s5i5": [_obs("s5i5", 0, 3, 3), _obs("s5i5", 1, 9, 3)],
    }


def _registry() -> dict[str, object]:
    return {
        "reproducible_total_levels": 177,
        "games": [
            {"game": "dc22", "reproducibility": "reproduced", "levels_reproduced": 6},
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 8},
            {"game": "s5i5", "reproducibility": "reproduced", "levels_reproduced": 8},
        ],
    }


def test_req_arc_wmte_5630_spec_declares_probe_contract() -> None:
    """REQ-ARC-WMTE-5630: OpenSpec pins provenance, object hypotheses, and fail-closed
    readiness before implementation code is allowed to satisfy the task."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5630") :]

    for marker in (
        "SCENARIO-ARC-WMTE-5630-INFORMATIVE-PROBE-POSITIVE",
        "SCENARIO-ARC-WMTE-5630-NEGATIVE-AND-UNSAFE-REJECTION",
        'solve_provenance="development_proxy"',
        "bounded_object_hypothesis_search_over_live_agent_observations",
        "unsafe_model_accept_count",
    ):
        assert marker in section


def test_scenario_5630_build_artifact_reports_ready_positive_controls() -> None:
    """SCENARIO-ARC-WMTE-5630-INFORMATIVE-PROBE-POSITIVE: synthetic reproduced-level
    traces exercise the full artifact gate without claiming a solve."""

    artifact = mod.build_artifact(
        transitions_by_game=_traces(),
        registry=_registry(),
        roster=("dc22", "bp35", "s5i5"),
        random_seed=5630,
    )

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["registry_precheck_receipt"]["ok"] is True
    assert artifact["registry_precheck_receipt"]["only_already_reproduced_levels"] is True
    assert artifact["evaluation_levels"] == ["dc22:L<=6", "bp35:L<=8", "s5i5:L<=8"]
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["live_observation_fields_used"] == [
        "state",
        "action",
        "data",
        "successor",
        "level_before",
        "level_after",
    ]
    assert artifact["object_hypothesis_non_degenerate_count"] == 3
    assert artifact["informative_control_delta"] > 0.0
    assert artifact["uninformative_control_delta"] <= 0.0
    assert artifact["unsafe_model_accept_count"] == 0
    assert artifact["live_interface_replay_rate"] == 1.0
    assert artifact["epistemic_probe_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]


def test_req_arc_wmte_5630_checked_in_artifact_has_required_schema() -> None:
    """REQ-ARC-WMTE-5630: the checked-in JSON is the stable experiment deliverable."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["registry_precheck_receipt"]["ok"] is True
    assert len(artifact["evaluation_levels"]) >= 3
    assert artifact["unsafe_model_accept_count"] == 0
    assert artifact["live_interface_replay_rate"] == 1.0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["honest_verdict"].startswith(("complete:", "blocked:"))
