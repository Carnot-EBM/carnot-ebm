"""Tests for Exp5619 ARC forward/inverse transition-cycle verifier artifact.

Spec refs: REQ-ARC-WMTE-5619,
SCENARIO-ARC-WMTE-5619-CYCLE-ADMISSION,
SCENARIO-ARC-WMTE-5619-CORRUPTION-REJECTION.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot import experiment_5619_arc_forward_inverse_transition_cycle as mod
from carnot.agentic.arc_transition_cycle_verifier import ObservedTransition


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _synthetic_game(game: str, count: int = 54) -> list[ObservedTransition]:
    rows: list[ObservedTransition] = []
    for step in range(count):
        y = 1 + ((step * 3) % 6)
        x = 1 + ((step * 5) % 6)
        before = np.zeros((9, 9), dtype=np.int16)
        before[y, x] = 1
        after = before.copy()
        after[y, x] = 5
        rows.append(
            ObservedTransition(
                game=game,
                episode=f"{game}-ep-{step // 9}",
                step=step,
                state=before,
                action=6,
                data={"x": x, "y": y},
                successor=after,
            )
        )
    for step in range(count, count * 2):
        y = 1 + ((step * 2) % 6)
        before = np.zeros((9, 9), dtype=np.int16)
        before[y, 1] = 2
        after = before.copy()
        after[y, 1] = 0
        after[y, 2] = 2
        rows.append(
            ObservedTransition(
                game=game,
                episode=f"{game}-key-{step // 9}",
                step=step,
                state=before,
                action=1,
                data=None,
                successor=after,
            )
        )
    return rows


def _registry() -> dict[str, object]:
    return {
        "reproducible_total_levels": 177,
        "games": [
            {"game": "ga", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "gb", "reproducibility": "reproduced", "levels_reproduced": 3},
            {"game": "gc", "reproducibility": "reproduced", "levels_reproduced": 4},
        ],
    }


def test_req_arc_wmte_5619_spec_declares_cycle_verifier_contract() -> None:
    """REQ-ARC-WMTE-5619: OpenSpec pins provenance, two-factor verification, and
    fail-closed admission before implementation code changes."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5619") :]

    for marker in (
        "SCENARIO-ARC-WMTE-5619-CYCLE-ADMISSION",
        "SCENARIO-ARC-WMTE-5619-CORRUPTION-REJECTION",
        "source_files_read=false",
        "per_game_adapter_used=false",
        "offline_arcade_live_agent_runtime_filters_no_new_llm",
        "unsafe_transition_accept_count",
    ):
        assert marker in section


def test_scenario_5619_build_artifact_reports_positive_and_corruption_controls() -> None:
    """SCENARIO-ARC-WMTE-5619-CORRUPTION-REJECTION: build_artifact freezes thresholds,
    evaluates valid/corrupt held-outs, and records immutable update receipts only for
    accepted valid transitions."""

    traces = {game: _synthetic_game(game) for game in ("ga", "gb", "gc")}
    artifact = mod.build_artifact(
        transitions_by_game=traces,
        registry=_registry(),
        random_seed=5619,
        heldout_per_game=36,
    )

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["registry_precheck"]["only_known_levels_used"] is True
    assert artifact["source_files_read"] is False
    assert artifact["per_game_adapter_used"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["heldout_transitions_by_condition"]["valid"] >= 32
    for condition in mod.CORRUPTION_CONDITIONS:
        assert artifact["heldout_transitions_by_condition"][condition] >= 32
    assert artifact["valid_transition_accept_rate"] > 0.5
    assert artifact["corruption_reject_rate"] == 1.0
    assert artifact["unsafe_transition_accept_count"] == 0
    assert artifact["cycle_verifier_positive_control_rate"] > 0.5
    assert artifact["abstention_rate"] < 0.5
    assert artifact["inverse_action_accuracy"] >= artifact["valid_transition_accept_rate"]
    assert artifact["forward_replay_error"] == 0.0
    assert len(artifact["immutable_update_receipts"]) == artifact["valid_transition_accept_count"]
    assert artifact["honest_verdict"].startswith("complete:")
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]


def test_req_arc_wmte_5619_checked_in_artifact_has_required_schema() -> None:
    """REQ-ARC-WMTE-5619: the checked-in artifact is the stable Exp5619 deliverable."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["source_files_read"] is False
    assert artifact["per_game_adapter_used"] is False
    assert artifact["transition_feature_contract"]["inputs"] == [
        "state",
        "action",
        "successor",
    ]
    assert len(artifact["trace_roster"]) >= 3
    assert artifact["heldout_transitions_by_condition"]["valid"] >= 32
    for condition in mod.CORRUPTION_CONDITIONS:
        assert artifact["heldout_transitions_by_condition"][condition] >= 32
    assert artifact["unsafe_transition_accept_count"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"].startswith("sha256:")
