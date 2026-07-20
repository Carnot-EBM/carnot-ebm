"""Tests for Exp5753 frozen generic causal primitive live-registry A/B.

Spec refs: REQ-ARC-WMTE-5753,
SCENARIO-ARC-WMTE-5753-GATE-AND-SELECTION,
SCENARIO-ARC-WMTE-5753-LIVE-REACHABILITY-AND-LEAK-CANARIES,
SCENARIO-ARC-WMTE-5753-FULL-REGISTRY-PAIRED-AB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_5753_arc_generic_primitive_live_registry_ab as mod
from carnot.agentic.arc_generic_causal_primitives import (
    BoundaryCollisionPrimitive,
    coerce_generic_causal_primitive,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
pytestmark = pytest.mark.memory_watchdog_skip


class _Frame:
    def __init__(self, grid: list[list[int]]) -> None:
        self.frame = np.array(grid, dtype=np.int16)


def _source_artifact() -> dict[str, Any]:
    return mod.read_json(REPO / mod.EXP5740_RELATIVE_PATH)


def _corrigendum_artifact() -> dict[str, Any]:
    return mod.read_json(REPO / mod.EXP5745_RELATIVE_PATH)


def _registry_fixture(levels: int = 183, games: int = 25) -> dict[str, Any]:
    return {
        "reproducible_total_games": games,
        "reproducible_total_levels": levels,
        "games": [
            {
                "game": f"g{i:02d}",
                "levels_reproduced": max(1, levels // max(1, games)),
                "full_game_clear": True,
                "reproducibility": "reproduced",
            }
            for i in range(games)
        ],
    }


def _row(game: str, *, arm: str, level: int, repeated: int = 0) -> dict[str, Any]:
    return {
        "game": game,
        "arm": arm,
        "seed": mod.RANDOM_SEEDS[0],
        "action_budget": mod.ACTION_BUDGET,
        "actions_used": 10 + repeated,
        "levels_reproduced": level,
        "action_effect_predictions": 8,
        "action_effect_correct": 4 + level,
        "valid_actions": 9,
        "invalid_actions": 1,
        "repeated_actions": repeated,
        "unique_states": 5 + level,
        "planning_reachable": level > 0,
        "planning_attempts": 1,
        "budget_exhausted": level == 0,
        "crashed": False,
        "duration_s": 0.05,
        "actions_per_reproduced_level": None if level == 0 else 10.0 + repeated,
        "receipts": [
            {
                "step": 0,
                "observation_hash": f"sha256:{game}{arm}".ljust(71, "0")[:71],
                "action": None,
                "reward": 0.0,
                "state_hash": f"sha256:{arm}{game}".ljust(71, "1")[:71],
            }
        ],
        "failed_reason": None,
    }


def _fake_pairs() -> dict[str, Any]:
    pairs = []
    for i in range(25):
        game = f"g{i:02d}"
        base_level = 1 if i < 4 else 0
        prim_level = 1 if i < 5 else 0
        pairs.append(
            {
                "game": game,
                "seed": mod.RANDOM_SEEDS[0],
                "baseline": _row(game, arm="baseline", level=base_level, repeated=2),
                "primitive": _row(game, arm="primitive", level=prim_level, repeated=1),
                "receipts_preserved": True,
                "failed_reason": None,
            }
        )
    return {"pairs": pairs, "duration_s": 1.25}


def test_req_arc_wmte_5753_spec_declares_live_registry_ab_contract() -> None:
    """REQ-ARC-WMTE-5753: OpenSpec lists the full A/B schema and principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5753") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "SCENARIO-ARC-WMTE-5753-GATE-AND-SELECTION",
        "SCENARIO-ARC-WMTE-5753-LIVE-REACHABILITY-AND-LEAK-CANARIES",
        "SCENARIO-ARC-WMTE-5753-FULL-REGISTRY-PAIRED-AB",
        "fixed 400-action budgets",
        "no public level is eligible for new credit",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_arc_wmte_5753_gate_and_selection_are_hash_linked() -> None:
    """SCENARIO-ARC-WMTE-5753-GATE-AND-SELECTION: gates precede primitive choice."""

    source = _source_artifact()
    corrigendum = _corrigendum_artifact()
    selected = mod.select_primitive_from_exp5740(source, corrigendum)
    precheck = mod.registry_precheck(_registry_fixture(), registry_hash="sha256:test")
    preconditions = mod.structured_preconditions(
        root=REPO,
        check_arc_environment=False,
        check_resources=False,
    )

    assert precheck["ok"] is True
    assert precheck["public_game_count"] == 25
    assert precheck["registry_level_count"] == 183
    assert precheck["all_public_games_complete"] is True
    assert precheck["no_public_level_can_be_credited_as_new"] is True
    assert selected["selected_primitive_id"] == "boundary_or_collision"
    assert selected["selected_primitive_hash"].startswith("sha256:")
    assert selected["selection_rule"] == mod.SELECTION_RULE
    assert preconditions["registry_precheck"]["public_game_count"] == 25
    assert preconditions["registry_precheck"]["registry_level_count"] == 183
    assert preconditions["upstream_gates"]["exp5745_normalized_gate_passed"] is True
    assert preconditions["ok"] is True


def test_scenario_arc_wmte_5753_blocked_gate_bails_before_live_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5753-GATE-AND-SELECTION: failed gates stop the A/B."""

    monkeypatch.setattr(
        mod,
        "structured_preconditions",
        lambda **_kw: {
            "ok": False,
            "failures": ["blocked_gate_check_failed"],
            "registry_precheck": mod.registry_precheck(_registry_fixture()),
            "upstream_artifact_hashes": {},
            "upstream_gates": {"exp5745_normalized_gate_passed": False},
        },
    )

    def _fail_if_live_called(*_args, **_kwargs):
        raise AssertionError("live A/B must not run after a failed structured gate")

    monkeypatch.setattr(mod, "run_matched_full_registry_ab", _fail_if_live_called)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "retired: blocked_gate_check_failed_repeated_no_gate_weakening"
    assert artifact["retirement_signal"] == "retire_generic_primitive_live_registry_ab"
    assert artifact["per_game_metrics"] == []
    assert artifact["baseline_live_levels_reproduced"] == 0
    assert artifact["primitive_live_levels_reproduced"] == 0
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_solve_credited"] is False
    assert set(artifact["field_principles"]) == set(artifact)


def test_scenario_arc_wmte_5753_live_reachability_and_canaries() -> None:
    """SCENARIO-ARC-WMTE-5753-LIVE-REACHABILITY-AND-LEAK-CANARIES."""

    primitive = BoundaryCollisionPrimitive()
    clean_rows = [{"action": 1, "data": None, "state_hash": "sha256:clean"}]
    rejected_rows = [
        {"action": 1, "game_id": "lp85"},
        {"action": 2, "source_file": "environment_files/lp85.py"},
    ]

    leak_receipt = primitive.game_blind_receipt(clean_rows + rejected_rows)

    assert mod.primitive_live_reachability_receipt(primitive)["primitive_live_reachable"] is True
    assert leak_receipt["admitted_source_leak_count"] == 0
    assert leak_receipt["admitted_game_identity_leak_count"] == 0
    assert leak_receipt["detected_source_leak_canary_count"] == 1
    assert leak_receipt["detected_game_identity_leak_canary_count"] == 1
    assert mod.static_leak_canaries()["admitted_source_leak_count"] == 0
    assert mod.static_leak_canaries()["admitted_game_identity_leak_count"] == 0
    assert mod.PRODUCTION_DEFAULT_ENABLED is False


def test_scenario_arc_wmte_5753_boundary_primitive_ranks_visible_no_effect_last() -> None:
    """SCENARIO-ARC-WMTE-5753-LIVE-REACHABILITY-AND-LEAK-CANARIES."""

    primitive = BoundaryCollisionPrimitive()
    before = _Frame([[0, 1], [0, 1]])
    after_same = _Frame([[0, 1], [0, 1]])
    after_changed = _Frame([[0, 2], [0, 1]])

    first_rank = primitive.rank_candidates(
        before,
        [{"action": 1, "data": None}, {"action": 2, "data": None}],
    )
    no_change_receipt = primitive.observe_transition(before, 1, None, after_same)
    changed_receipt = primitive.observe_transition(before, 2, None, after_changed)
    second_rank = primitive.rank_candidates(
        before,
        [{"action": 1, "data": None}, {"action": 2, "data": None}],
    )

    assert first_rank[0]["action"] == 1
    assert no_change_receipt["rank_after_threshold"] is True
    assert changed_receipt["rank_after_threshold"] is False
    assert second_rank[0]["action"] == 2
    assert second_rank[1]["action"] == 1
    assert primitive.diagnostics()["blocked_signature_count"] == 1
    assert coerce_generic_causal_primitive(True).primitive_id == "boundary_or_collision"
    assert coerce_generic_causal_primitive("boundary_or_collision").primitive_id == (
        "boundary_or_collision"
    )
    assert coerce_generic_causal_primitive(False) is None
    assert coerce_generic_causal_primitive(primitive) is primitive


def test_scenario_arc_wmte_5753_full_registry_paired_ab_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5753-FULL-REGISTRY-PAIRED-AB: aggregate schema is complete."""

    monkeypatch.setattr(
        mod,
        "structured_preconditions",
        lambda **_kw: {
            "ok": True,
            "failures": [],
            "registry_precheck": mod.registry_precheck(_registry_fixture()),
            "upstream_artifact_hashes": {"exp5740": "sha256:0"},
            "upstream_gates": {"exp5745_normalized_gate_passed": True},
            "arc_environment": {"reachable": True},
            "resource_precheck": {"ok": True},
            "submitted_live_policy_path": {"reachable": True},
        },
    )
    monkeypatch.setattr(mod, "run_matched_full_registry_ab", lambda **_kw: _fake_pairs())

    artifact = mod.build_artifact(
        root=tmp_path,
        test_commands=["unit: exp5753"],
        test_exit_codes={"unit: exp5753": 0},
    )
    saved = mod.write_output(tmp_path, artifact)

    assert json.loads(saved.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["public_game_count"] == 25
    assert artifact["registry_level_count"] == 183
    assert artifact["selected_primitive_id"] == "boundary_or_collision"
    assert artifact["primitive_live_reachable"] is True
    assert artifact["source_leak_count"] == 0
    assert artifact["game_identity_leak_count"] == 0
    assert artifact["paired_trial_manifest"]["game_count"] == 25
    assert artifact["paired_trial_manifest"]["action_budget"] == 400
    assert len(artifact["per_game_metrics"]) == 25
    assert artifact["baseline_live_levels_reproduced"] == 4
    assert artifact["primitive_live_levels_reproduced"] == 5
    assert artifact["live_level_reproduction_delta"] == 1
    assert artifact["repeated_action_rate_delta"] < 0.0
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_solve_credited"] is False
    assert artifact["outer_loop_re_used"] is False
    assert artifact["per_game_adapter_used"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_arc_wmte_5753_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-5753: checked-in JSON is the stable terminal receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_solve_credited"] is False
    assert artifact["public_game_count"] == 25
    assert artifact["registry_level_count"] == 183
    assert artifact["selected_primitive_id"] == "boundary_or_collision"
    assert len(artifact["per_game_metrics"]) in {0, 25}
