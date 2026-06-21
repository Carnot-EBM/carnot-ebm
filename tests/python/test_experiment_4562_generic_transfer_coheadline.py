"""Tests for Exp 4562 generic-transfer co-headline.

Spec refs: REQ-CAPSTONE-4562, SCENARIO-CAPSTONE-4562,
SCENARIO-CAPSTONE-4562-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4562_generic_transfer_coheadline as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_registry(root: Path, total: int) -> None:
    registry = root / mod.exp4550.REGISTRY_RELATIVE_PATH
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
    solved_signatures = {"g1~color01", "g2~color02"}
    solved = str(spec["variant_signature"]) in solved_signatures
    reached = 1 if solved else 0
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": solved,
        "reached_level": reached,
        "actions": 5,
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
        "actions": 3,
        "reproduction_gate": {
            "game": game,
            "claimed_level": 0,
            "reached_level": 0,
            "reproduced": False,
        },
        "blocked_reason": "",
    }


def test_req_capstone_4562_spec_declares_coheadline_contract() -> None:
    """REQ-CAPSTONE-4562: OpenSpec declares the co-headline schema before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4562" in spec
    assert "SCENARIO-CAPSTONE-4562" in spec
    assert "SCENARIO-CAPSTONE-4562-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4562_reports_bank_transfer_and_ci(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4562: small variants compute both co-headlines and a bracketing CI."""

    games = ("g1", "g2")
    _write_registry(tmp_path, total=52)

    artifact = mod.run(
        root=tmp_path,
        public_games=games,
        variant_ids=(1, 2, 3),
        budget=12,
        preconditions_checked=_preconditions(games),
        variant_runner=_fake_variant_runner,
        n_bootstrap=80,
        write=True,
    )

    assert artifact["honest_verdict"] == "shipped: generic_transfer_coheadline_with_ci_wired"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducible_total_levels"] == 52
    assert artifact["variant_attempts_count"] == 6
    assert artifact["variant_solved_count"] == 2
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(2 / 6)
    assert 0.0 <= artifact["generic_transfer_rate_over_variants"] <= 1.0
    assert artifact["generic_transfer_ci"][0] <= artifact["generic_transfer_rate_over_variants"]
    assert artifact["generic_transfer_ci"][1] >= artifact["generic_transfer_rate_over_variants"]
    assert artifact["variant_plan"]["variants_per_game"] == 3
    assert artifact["metric_wired_into_capstone"]["reported_side_by_side"] == [
        "reproducible_total_levels",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
    ]
    assert artifact["metric_wired_into_capstone"]["known_game_bank_inflates_transfer"] is False
    assert "KNOWN-game solve capability" in artifact["honest_metric_framing"]
    assert "real leaderboard signal" in artifact["honest_metric_framing"]
    assert artifact["leaderboard_submission"] is False
    assert artifact["tests_added_pass"]["passed"] is True
    assert mod.validate_artifact(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_capstone_4562_known_game_only_bank_cannot_inflate_transfer(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4562: a large known-game bank does not raise held-out transfer."""

    games = ("g1", "g2", "g3")
    _write_registry(tmp_path, total=123)

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1, 2),
        budget=12,
        preconditions_checked=_preconditions(games),
        variant_runner=_no_transfer_runner,
        n_bootstrap=40,
    )

    assert artifact["reproducible_total_levels"] == 123
    assert artifact["variant_attempts_count"] == 6
    assert artifact["variant_solved_count"] == 0
    assert artifact["generic_transfer_rate_over_variants"] == 0.0
    assert artifact["generic_transfer_ci"] == [0.0, 0.0]
    assert artifact["metric_wired_into_capstone"]["known_game_bank_inflates_transfer"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4562_partial_when_precondition_missing(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4562: missing variant support yields a terminal partial artifact."""

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

    assert artifact["honest_verdict"] == "complete: generic_transfer_coheadline_partial_variant_flag"
    assert artifact["reproducible_total_levels"] == 7
    assert artifact["generic_transfer_rate_over_variants"] == 0.0
    assert artifact["generic_transfer_ci"] == [0.0, 0.0]
    assert artifact["variant_attempts"] == []
    assert mod.validate_artifact(artifact) == []
