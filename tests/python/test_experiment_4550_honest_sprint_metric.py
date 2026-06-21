"""Tests for Exp 4550 honest sprint metric wiring.

Spec refs: REQ-CAPSTONE-4550, SCENARIO-CAPSTONE-4550,
SCENARIO-CAPSTONE-4550-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4550_honest_sprint_metric as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_registry(root: Path, total: int) -> None:
    registry = root / mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        f"schema_version: 1\nreproducible_total_levels: {total}\n",
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
    }


def _fake_variant_runner(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
    solved = game == "g1" and int(spec["variant"]) == 1
    reached = 1 if solved else 0
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": spec["variant"],
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": solved,
        "reached_level": reached,
        "actions": 4,
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
        "variant": spec["variant"],
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": False,
        "reached_level": 0,
        "actions": 2,
        "reproduction_gate": {
            "game": game,
            "claimed_level": 0,
            "reached_level": 0,
            "reproduced": False,
        },
        "blocked_reason": "",
    }


def test_req_capstone_4550_spec_declares_honest_sprint_metric_contract() -> None:
    """REQ-CAPSTONE-4550: OpenSpec declares both side-by-side sprint metrics."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4550" in spec
    assert "SCENARIO-CAPSTONE-4550" in spec
    assert "SCENARIO-CAPSTONE-4550-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_capstone_4550_computes_bank_and_variant_transfer(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4550: a small variant set computes both metrics and bounds the rate."""

    games = ("g1", "g2")
    _write_registry(tmp_path, total=51)

    artifact = mod.run(
        root=tmp_path,
        public_games=games,
        variant_ids=(1, 2),
        preconditions_checked=_preconditions(games),
        variant_runner=_fake_variant_runner,
        write=True,
    )

    assert artifact["honest_verdict"] == "shipped: honest_sprint_metric_variant_transfer_wired"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducible_total_levels"] == 51
    assert artifact["variant_attempts_count"] == 4
    assert artifact["variant_solved_count"] == 1
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(0.25)
    assert 0.0 <= artifact["generic_transfer_rate_over_variants"] <= 1.0
    assert artifact["metric_wired_into_capstone"]["reported_side_by_side"] == [
        "reproducible_total_levels",
        "generic_transfer_rate_over_variants",
    ]
    assert "KNOWN games" in artifact["honest_metric_framing"]
    assert "held-out-proxy generalization" in artifact["honest_metric_framing"]
    assert artifact["tests_added_pass"]["passed"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_capstone_4550_known_game_bank_does_not_inflate_transfer_rate(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4550: known-game bank count is reported but cannot raise variant transfer."""

    games = ("g1", "g2", "g3")
    _write_registry(tmp_path, total=99)

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        preconditions_checked=_preconditions(games),
        variant_runner=_no_transfer_runner,
    )

    assert artifact["reproducible_total_levels"] == 99
    assert artifact["variant_attempts_count"] == 3
    assert artifact["variant_solved_count"] == 0
    assert artifact["generic_transfer_rate_over_variants"] == 0.0
    assert artifact["metric_wired_into_capstone"]["known_game_bank_inflates_transfer"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4550_partial_when_variant_precondition_missing(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4550: missing shipped variant flags yields a terminal partial artifact."""

    games = ("g1",)
    _write_registry(tmp_path, total=7)
    preconditions = _preconditions(games)
    preconditions["variant_flag_present"] = False
    preconditions["ok"] = False

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        preconditions_checked=preconditions,
        variant_runner=_fake_variant_runner,
    )

    assert artifact["honest_verdict"] == "complete: honest_sprint_metric_partial_variant_flag"
    assert artifact["reproducible_total_levels"] == 7
    assert artifact["generic_transfer_rate_over_variants"] == 0.0
    assert artifact["variant_attempts"] == []
    assert artifact["preconditions_checked"]["variant_flag_present"] is False
    assert mod.validate_artifact(artifact) == []
