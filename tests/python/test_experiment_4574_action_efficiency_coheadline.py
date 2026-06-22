"""Tests for Exp 4574 action-efficiency co-headline.

Spec refs: REQ-CAPSTONE-4574, SCENARIO-CAPSTONE-4574,
SCENARIO-CAPSTONE-4574-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4574_action_efficiency_coheadline as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_registry(root: Path, total: int) -> None:
    registry = root / mod.exp4550.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        f"schema_version: 1\nreproducible_total_levels: {total}\n",
        encoding="utf-8",
    )


def _write_human_replay_shard(root: Path) -> None:
    corpus = root / mod.HUMAN_REPLAY_RELATIVE_PATH
    shard = corpus / "shards" / "train-00000.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"env": "g1", "guid": "h1", "step_index": 1, "level_progress": 0.0},
        {"env": "g1", "guid": "h1", "step_index": 5, "level_progress": 0.5},
        {"env": "g1", "guid": "h1", "step_index": 6, "level_progress": 0.5},
        {"env": "g2", "guid": "h2", "step_index": 2, "level_progress": 0.0},
        {"env": "g2", "guid": "h2", "step_index": 3, "level_progress": 0.25},
    ]
    shard.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "carnot.arc_human_replay.frame_action_delta.v1",
                "example_count": len(rows),
                "shard_count": 1,
                "shards": [{"path": "shards/train-00000.jsonl", "rows": len(rows)}],
                "source_metadata": {"test": True},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _preconditions(games: tuple[str, ...]) -> dict[str, Any]:
    return {
        "ok": True,
        "arc_leaderboard_eval_import": True,
        "variant_flag_present": True,
        "reflect_flag_present": True,
        "offline_env_public_games": list(games),
        "variant_env_import": True,
        "leaderboard_submission": False,
        "exp4550_measure_generic_transfer_over_variants_import": True,
    }


def _variant_runner(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
    solved_actions = {"g1~color01": 8, "g2~color01": 10}
    signature = str(spec["variant_signature"])
    solved = signature in solved_actions
    reached = 1 if solved else 0
    return {
        "game": game,
        "variant_signature": signature,
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": solved,
        "reached_level": reached,
        "actions": int(solved_actions.get(signature, 99)),
        "actions_to_first_levelup": solved_actions.get(signature),
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached,
            "reached_level": reached,
            "reproduced": solved,
        },
        "blocked_reason": "",
    }


def _no_transfer_runner(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": False,
        "reached_level": 0,
        "actions": 99,
        "actions_to_first_levelup": None,
        "reproduction_gate": {
            "game": game,
            "claimed_level": 0,
            "reached_level": 0,
            "reproduced": False,
        },
        "blocked_reason": "",
    }


def test_req_capstone_4574_spec_declares_action_efficiency_coheadline() -> None:
    """REQ-CAPSTONE-4574: OpenSpec declares the third co-headline metric."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4574" in spec
    assert "SCENARIO-CAPSTONE-4574" in spec
    assert "SCENARIO-CAPSTONE-4574-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4574_computes_three_metrics_and_ci(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4574: variants and replay rows compute all co-headlines."""

    games = ("g1", "g2")
    _write_registry(tmp_path, total=52)
    _write_human_replay_shard(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        public_games=games,
        variant_ids=(1, 2),
        budget=12,
        preconditions_checked=_preconditions(games),
        variant_runner=_variant_runner,
        n_bootstrap=80,
        write=True,
    )

    assert artifact["honest_verdict"] == "shipped: action_efficiency_coheadline_with_ci_wired"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducible_total_levels"] == 52
    assert artifact["variant_attempts_count"] == 4
    assert artifact["variant_solved_count"] == 2
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(0.5)
    assert artifact["generic_transfer_ci"][0] <= artifact["generic_transfer_rate_over_variants"]
    assert artifact["generic_transfer_ci"][1] >= artifact["generic_transfer_rate_over_variants"]
    assert artifact["median_actions_to_first_levelup"] == pytest.approx(9.0)
    assert artifact["human_baseline_actions"] == pytest.approx(4.0)
    assert artifact["agent_actions_to_first_levelup"] == [8, 10]
    assert artifact["human_baseline_sample_count"] == 2
    assert artifact["action_efficiency_score"] == pytest.approx((4.0 / 9.0) ** 2)
    assert 0.0 <= artifact["action_efficiency_score"] <= 1.0
    assert artifact["action_efficiency_ci"][0] <= artifact["action_efficiency_score"]
    assert artifact["action_efficiency_ci"][1] >= artifact["action_efficiency_score"]
    assert artifact["metric_wired_into_capstone"]["reported_side_by_side"] == [
        "reproducible_total_levels",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
        "action_efficiency_score",
        "action_efficiency_ci",
    ]
    assert artifact["metric_wired_into_capstone"]["known_game_bank_inflates_transfer"] is False
    assert artifact["metric_wired_into_capstone"]["known_game_bank_inflates_action_efficiency"] is False
    assert "bank count = KNOWN-game capability" in artifact["honest_metric_framing"]
    assert "action efficiency = the literal leaderboard scoring term" in artifact["honest_metric_framing"]
    assert artifact["leaderboard_submission"] is False
    assert artifact["tests_added_pass"]["passed"] is True
    assert mod.validate_artifact(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_capstone_4574_known_game_bank_cannot_inflate_action_efficiency(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4574: known-game banks do not raise held-out action efficiency."""

    games = ("g1", "g2", "g3")
    _write_registry(tmp_path, total=999)
    _write_human_replay_shard(tmp_path)

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1, 2),
        budget=12,
        preconditions_checked=_preconditions(games),
        variant_runner=_no_transfer_runner,
        n_bootstrap=40,
    )

    assert artifact["reproducible_total_levels"] == 999
    assert artifact["variant_attempts_count"] == 6
    assert artifact["variant_solved_count"] == 0
    assert artifact["generic_transfer_rate_over_variants"] == 0.0
    assert artifact["generic_transfer_ci"] == [0.0, 0.0]
    assert artifact["agent_actions_to_first_levelup"] == []
    assert artifact["median_actions_to_first_levelup"] is None
    assert artifact["action_efficiency_score"] == 0.0
    assert artifact["action_efficiency_ci"] == [0.0, 0.0]
    assert artifact["metric_wired_into_capstone"]["known_game_bank_inflates_action_efficiency"] is False
    assert mod.validate_artifact(artifact) == []
