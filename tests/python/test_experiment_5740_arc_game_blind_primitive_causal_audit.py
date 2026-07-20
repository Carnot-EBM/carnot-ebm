"""Tests for Exp5740 ARC game-blind primitive causal audit.

Spec refs: REQ-ARC-WMTE-5740,
SCENARIO-ARC-WMTE-5740-IDENTITY-AND-SOURCE-LEAK-REJECTION,
SCENARIO-ARC-WMTE-5740-DELETION-REPLAY-CAUSAL-UTILITY,
SCENARIO-ARC-WMTE-5740-DIAGNOSTIC-NO-CREDIT-CONTRACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5740_arc_game_blind_primitive_causal_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _sha(label: str) -> str:
    return "sha256:" + (label.encode("utf-8").hex() * 4)[:64].ljust(64, "0")


def _frame(
    frame_index: int,
    grid_hash: str,
    *,
    move: dict[str, Any],
    colors: list[int] | None = None,
    level: int = 0,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "action_count": frame_index,
        "levels_completed": level,
        "grid_shape": [8, 8],
        "grid_hash": _sha(grid_hash),
        "colors": colors or [0, 1, 2],
        "available_actions": ["1", "2", "3", "4", "5", "6", "7"],
        "move": move,
    }


def _frame_sequence(game_index: int) -> list[dict[str, Any]]:
    prefix = f"g{game_index:02d}"
    return [
        _frame(0, f"{prefix}-a", move={"kind": "RESET", "data": None}),
        _frame(1, f"{prefix}-b", move={"kind": 1, "data": None}),
        _frame(2, f"{prefix}-a", move={"kind": 2, "data": None}),
        _frame(3, f"{prefix}-a", move={"kind": 5, "data": None}),
        _frame(4, f"{prefix}-a", move={"kind": 5, "data": None}),
        _frame(5, f"{prefix}-a", move={"kind": 5, "data": None}),
        _frame(6, f"{prefix}-c", move={"kind": 6, "data": {"x": 1, "y": 1}}, colors=[0, 1, 3]),
        _frame(7, f"{prefix}-c", move={"kind": 4, "data": None}, colors=[0, 1, 3]),
        _frame(8, f"{prefix}-d", move={"kind": 4, "data": None}, colors=[0, 1, 3]),
    ]


def _write_fixture_repo(root: Path) -> Path:
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    registry_rows = [
        {
            "game": f"g{i:02d}",
            "levels_reproduced": 1,
            "full_game_clear": True,
            "reproducibility": "reproduced",
        }
        for i in range(25)
    ]
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reproducible_total_games": 25,
                "games": registry_rows,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    per_game = []
    for i in range(25):
        per_game.append(
            {
                "game": f"g{i:02d}",
                "levels": 0,
                "actions": 8,
                "frame_sequence": _frame_sequence(i),
                "policy_diagnostics": {
                    "induction_attempts": [
                        {
                            "reason": "stall",
                            "transition_count": 8,
                            "planned": False,
                            "skipped": "world_model_accuracy_below_threshold",
                        }
                    ],
                    "decision_hash": _sha(f"decision-{i}"),
                },
            }
        )
    (root / mod.LIVE_GAP_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "experiment": "arc_live_oracle_gap",
                "games_mode": "oracle",
                "policy": "e3",
                "budget": 400,
                "per_game": per_game,
            }
        ),
        encoding="utf-8",
    )
    (root / "results/experiment_5727_arc_generalization_live_oracle_gap_v511.json").write_text(
        json.dumps({"experiment": "fixture", "games_measured": 25}),
        encoding="utf-8",
    )
    (root / "results/experiment_5727_perception_action_effect_adequacy.json").write_text(
        json.dumps({"experiment": "fixture", "target": "action_effect"}),
        encoding="utf-8",
    )
    return root


def test_req_arc_wmte_5740_spec_declares_causal_audit_contract() -> None:
    """REQ-ARC-WMTE-5740: OpenSpec anchors fields and leak-control scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5740") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]

    for marker in (
        "SCENARIO-ARC-WMTE-5740-IDENTITY-AND-SOURCE-LEAK-REJECTION",
        "SCENARIO-ARC-WMTE-5740-DELETION-REPLAY-CAUSAL-UTILITY",
        "SCENARIO-ARC-WMTE-5740-DIAGNOSTIC-NO-CREDIT-CONTRACT",
        str(mod.RESULT_RELATIVE_PATH),
        "object_displacement",
        "future-frame leak",
        "no-op deletion",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_arc_wmte_5740_leak_controls_are_detected_and_rejected(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5740-IDENTITY-AND-SOURCE-LEAK-REJECTION."""

    root = _write_fixture_repo(tmp_path)
    artifact = mod.build_artifact(root=root)

    mod.validate_artifact(artifact)
    controls = {row["control"]: row for row in artifact["negative_controls"]}
    for name in (
        "game_id_leak",
        "per_game_constant",
        "future_frame_leak",
        "source_derived_rule",
    ):
        assert controls[name]["detected"] is True
        assert controls[name]["rejected"] is True

    assert artifact["source_leak_count"] == 1
    assert artifact["game_identity_leak_count"] == 2
    assert artifact["game_identity_stripping_receipts"]
    for receipt in artifact["game_identity_stripping_receipts"]:
        assert {"game", "game_id", "game_name"}.isdisjoint(receipt["after_keys"])
        assert receipt["stripped_keys"]


def test_scenario_arc_wmte_5740_deletion_replay_counts_only_causal_retention(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5740-DELETION-REPLAY-CAUSAL-UTILITY."""

    root = _write_fixture_repo(tmp_path)
    artifact = mod.build_artifact(root=root)

    mod.validate_artifact(artifact)
    assert artifact["counterfactual_receipt_coverage"]["paired_replay_count"] >= 30
    assert artifact["counterfactual_receipt_coverage"]["meets_minimum_n"] is True
    assert len(artifact["leave_one_game_out_splits"]) == 25
    assert set(artifact["primitive_schema"]) == set(mod.PRIMITIVE_FAMILIES)

    retained = [row for row in artifact["primitive_candidates"] if row["causal_retained"]]
    assert artifact["positive_causal_primitive_count"] == len(retained)
    assert artifact["positive_causal_primitive_count"] > 0
    for row in retained:
        assert row["paired_replay_count"] >= mod.MIN_PAIRED_REPLAYS
        assert row["corrected_interval"][0] > 0.0
        assert row["static_frequency_only"] is False
        assert row["retained_in_heldout_game_count"] >= mod.MIN_RETAINED_HELDOUT_GAMES

    utility_by_primitive = artifact["counterfactual_trajectory_utility"]
    for row in retained:
        utility = utility_by_primitive[row["primitive"]]
        assert utility["downstream_decision_hash_changed_count"] > 0
        assert utility["composite_utility_delta"] == row["composite_utility_delta"]


def test_scenario_arc_wmte_5740_checked_in_artifact_is_no_credit_diagnostic() -> None:
    """SCENARIO-ARC-WMTE-5740-DIAGNOSTIC-NO-CREDIT-CONTRACT."""

    path = REPO / mod.RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["registry_game_count"] == 25
    assert artifact["policy_modified"] is False
    assert artifact["registry_modified"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["registry_game_count_is_25"] is True
    assert artifact["honest_verdict"].startswith("complete:")
