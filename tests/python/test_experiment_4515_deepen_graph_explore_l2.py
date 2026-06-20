"""Tests for Exp 4515 graph-explore m0r0 L1->L2 deepening.

Spec refs: REQ-ARC-WMTE-4515, SCENARIO-ARC-WMTE-4515.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4515_deepen_graph_explore_l2 as exp4515
from carnot.agentic import arc_m0r0_adapter_logic as m0r0
from carnot.agentic.arc_game_adapters import adaptered_games, get_adapter


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


class FakeLevel:
    def __init__(self, sprites: list[SimpleNamespace], grid_size: tuple[int, int] = (5, 3)):
        self._sprites = sprites
        self.grid_size = grid_size

    def get_sprites_by_name(self, name: str) -> list[SimpleNamespace]:
        return [sprite for sprite in self._sprites if sprite.name == name]

    def get_sprites_by_tag(self, tag: str) -> list[SimpleNamespace]:
        return [sprite for sprite in self._sprites if tag in getattr(sprite, "tags", [])]


def _sprite(
    name: str,
    x: int,
    y: int,
    *,
    tags: tuple[str, ...] = (),
    pixels: tuple[tuple[int, ...], ...] = ((10,),),
    interaction: str = "ACTIVE",
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        x=x,
        y=y,
        tags=list(tags),
        pixels=[list(row) for row in pixels],
        rotation=0,
        visible=True,
        interaction=interaction,
    )


def _fake_game(*sprites: SimpleNamespace, grid_size: tuple[int, int] = (5, 3)) -> SimpleNamespace:
    return SimpleNamespace(
        current_level=FakeLevel(list(sprites), grid_size=grid_size),
        okpvcjupabr=set(),
        ukempikfmtm=-1,
    )


def _preconditions() -> dict[str, object]:
    return {
        "offline_arcade_import_smoke": True,
    }


def test_req_arc_wmte_4515_spec_declares_graph_explore_l2_contract() -> None:
    """REQ-ARC-WMTE-4515: OpenSpec names the 4515 graph-explore artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in ("REQ-ARC-WMTE-4515", "SCENARIO-ARC-WMTE-4515"):
        assert ref in spec
    assert exp4515.RESULT_RELATIVE_PATH in spec
    for phrase in (
        "su15",
        "sp80",
        "cn04",
        "m0r0",
        "sk48",
        "branch_mode='fresh_env'",
        "OfflineSolver.solve_level",
        "offline_reproduced=true",
        "reproduced_levels=2",
        "reproducibility_checksum",
    ):
        assert phrase in spec
    for field, principle in exp4515.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4515_m0r0_adapter_is_registered_for_fresh_env_routing() -> None:
    """REQ-ARC-WMTE-4515: m0r0 has a registered fresh-env GameAdapter."""

    adapter = get_adapter(exp4515.TARGET_GAME)

    assert exp4515.TARGET_GAME == "m0r0"
    assert exp4515.TARGET_GAME in adaptered_games()
    assert adapter is not None
    assert adapter.game == "m0r0"
    assert callable(adapter.action_labels)
    assert callable(adapter.apply)
    assert callable(adapter.state_key)
    assert callable(adapter.hand_verifier)
    assert adapter.branch_mode == "fresh_env"


def test_scenario_arc_wmte_4515_visible_m0r0_predicate_routes_around_hazards() -> None:
    """SCENARIO-ARC-WMTE-4515: L2 predicate is derived from visible sprites."""

    left = _sprite("pikgci-toljda-leklkn", 1, 1)
    right = _sprite("pikgci-toljda-rivmdg", 3, 1)
    hazard = _sprite("spswjz", 2, 1, tags=("spswjz",), pixels=((8,),))
    game = _fake_game(left, right, hazard)

    assert m0r0.m0r0_active_players(game) == (
        ("pikgci-toljda-leklkn", 1, 1),
        ("pikgci-toljda-rivmdg", 3, 1),
    )
    assert m0r0.m0r0_bad_cells(game) == ((2, 1),)
    assert m0r0.m0r0_visible_plan_actions(game) == [1, 4]
    assert m0r0.m0r0_hand_verifier(game) == 2.0
    assert m0r0.m0r0_action_labels(SimpleNamespace(_game=game)) == [
        json.dumps({"action": 1}, sort_keys=True, separators=(",", ":"))
    ]

    coalesced = _fake_game(
        _sprite("pikgci-toljda-leklkn", 2, 0),
        _sprite("pikgci-toljda-rivmdg", 2, 0),
    )
    assert m0r0.m0r0_visible_plan_actions(coalesced) == []
    assert m0r0.m0r0_hand_verifier(coalesced) == 0.0


def test_req_arc_wmte_4515_m0r0_state_key_includes_visible_blockers_and_hazards() -> None:
    """REQ-ARC-WMTE-4515: dedup state is anchored in visible env state."""

    left = _sprite("pikgci-toljda-leklkn", 1, 1)
    right = _sprite("pikgci-toljda-rivmdg", 3, 1)
    hazard = _sprite("spswjz", 2, 1, tags=("spswjz",), pixels=((8,),))
    wall = _sprite("wahtyt-LevelX", 0, 0, tags=("wahtyt",), pixels=((-1, 0),))
    key = m0r0.m0r0_state_key(_fake_game(left, right, hazard, wall))

    assert ("players", (("pikgci-toljda-leklkn", 1, 1), ("pikgci-toljda-rivmdg", 3, 1))) in key
    assert ("bad", ((2, 1),)) in key
    assert ("walls", ((1, 0),)) in key
    assert ("lose_animation", -1) in key


def test_req_arc_wmte_4515_success_artifact_requires_banked_l2_reproduction(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4515: success requires offline reproduction and banked level 2."""

    artifact = exp4515.build_artifact(
        preconditions_checked=_preconditions(),
        target_game="m0r0",
        adapter_registered=True,
        solution_labels=[json.dumps({"action": 4})],
        l1_prefix_labels=[json.dumps({"action": 1})],
        solve_reached_level=2,
        reproduction_gate={"game": "m0r0", "reached_level": 2, "reproduced": True},
        depth_cap=80,
        states_expanded=12,
        tests_pass=True,
        adapter_branch_mode="fresh_env",
    )

    assert artifact["experiment"] == "experiment_4515_deepen_graph_explore_l2"
    assert artifact["schema"] == "carnot.graph_explore_deepen_l2_4515.v1"
    assert artifact["spec_refs"] == exp4515.SPEC_REFS
    assert artifact["honest_verdict"] == "success: m0r0_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["l2_extension_labels"] == [json.dumps({"action": 4})]
    assert artifact["schema_errors"] == []
    assert exp4515.artifact_schema_errors(artifact) == []

    out = exp4515.write_artifact(artifact, root=tmp_path)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    fabricated = dict(artifact)
    fabricated["offline_reproduced"] = False
    assert any("success artifact" in error for error in exp4515.artifact_schema_errors(fabricated))
    with pytest.raises(ValueError, match="success artifact"):
        exp4515.write_artifact(fabricated, root=tmp_path)

    mutations = [
        (lambda item: item.__setitem__("experiment", "experiment_4504_adapter_deepen_l2"), "experiment"),
        (lambda item: item.__setitem__("schema", "bad"), "schema"),
        (lambda item: item.__setitem__("spec_refs", []), "spec_refs"),
        (lambda item: item.__setitem__("target_game", "ar25"), "HUD-register-stall"),
        (lambda item: item.__setitem__("adapter_branch_mode", "replay"), "fresh_env"),
        (lambda item: item.__setitem__("reproducibility_checksum", "bad"), "checksum"),
    ]
    for mutate, expected in mutations:
        changed = dict(artifact)
        mutate(changed)
        assert any(expected in error for error in exp4515.artifact_schema_errors(changed))


def test_scenario_arc_wmte_4515_runner_writes_injected_m0r0_l2_success(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4515: injected solve/replay writes stable success JSON."""

    fake_adapter = SimpleNamespace(
        game="m0r0",
        apply=lambda env, label, frame: frame,
        warmup_label=None,
        depth_caps={2: 80},
        branch_mode="fresh_env",
    )
    calls: list[tuple[str, int, int]] = []

    def fake_solver_runner(game: str, adapter: object, target_level: int, depth_cap: int):
        calls.append((game, target_level, depth_cap))
        return [json.dumps({"action": 1})], [json.dumps({"action": 4})], 2, 7

    def fake_reproduction_runner(
        game: str,
        labels: list[str],
        apply_fn: object,
        *,
        warmup_label: str | None,
        claimed_level: int,
    ) -> dict[str, object]:
        assert labels == [json.dumps({"action": 1}), json.dumps({"action": 4})]
        return {
            "game": game,
            "claimed_level": claimed_level,
            "reached_level": claimed_level,
            "reproduced": True,
        }

    artifact = exp4515.run_experiment(
        root=tmp_path,
        adapter_lookup=lambda game: fake_adapter,
        solver_runner=fake_solver_runner,
        reproduction_runner=fake_reproduction_runner,
        preconditions_checked=_preconditions(),
        tests_pass=True,
    )
    written = json.loads((tmp_path / exp4515.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert calls == [("m0r0", 2, 80)]
    assert artifact == written
    assert artifact["honest_verdict"] == "success: m0r0_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["adapter_registered"] is True


def test_scenario_arc_wmte_4515_runner_reports_honest_residual(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4515: blocked L2 emits residual, not fabricated success."""

    artifact = exp4515.run_experiment(
        root=tmp_path,
        adapter_lookup=lambda game: None,
        preconditions_checked=_preconditions(),
        tests_pass=True,
    )

    assert artifact["honest_verdict"] == "complete: m0r0_l2_honest_residual"
    assert artifact["adapter_registered"] is False
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 1
    assert artifact["residual_blockers"] == [
        "m0r0_adapter_not_registered",
        "m0r0_solver_reached_level_1",
        "m0r0_l2_not_reproduced",
    ]
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4515_runner_rejects_missing_resource_preconditions() -> None:
    """REQ-ARC-WMTE-4515: missing offline Arcade precondition blocks before replay."""

    with pytest.raises(RuntimeError, match="blocked_offline_arcade_import_smoke"):
        exp4515.ensure_preconditions_ready({"offline_arcade_import_smoke": False})
