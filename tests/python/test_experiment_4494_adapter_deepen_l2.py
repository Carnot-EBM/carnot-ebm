"""Tests for Exp 4494 adapter-routed ARC L1->L2 deepening.

Spec refs: REQ-ARC-WMTE-4496, SCENARIO-ARC-WMTE-4495.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_4494_adapter_deepen_l2 as exp4494
from carnot.agentic import arc_cd82_adapter_logic as cd82
from carnot.agentic.arc_exp4024_fifth_game_explore_first import apply_cd82_region_fill
from carnot.agentic.arc_game_adapters import adaptered_games, get_adapter


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions() -> dict[str, object]:
    return {
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "test",
    }


class _Sprite:
    def __init__(
        self,
        name: str,
        x: int,
        y: int,
        pixels: np.ndarray | None = None,
        *,
        center_color: int | None = None,
    ) -> None:
        self.name = name
        self.x = x
        self.y = y
        if pixels is not None:
            self.pixels = np.array(pixels, dtype=np.int16, copy=True)
        else:
            self.pixels = np.zeros((5, 5), dtype=np.int16)
            if center_color is not None:
                self.pixels[2, 2] = int(center_color)


class _Level:
    def __init__(self, canvas: np.ndarray, target: np.ndarray, palette: list[_Sprite]) -> None:
        self.canvas = _Sprite("xytrjjbyib", 0, 0, canvas)
        self.target = _Sprite("eoqnvkspoa-test", 30, 0, target)
        self.palette = list(palette)

    def get_sprites_by_name(self, name: str) -> list[_Sprite]:
        if name == "xytrjjbyib":
            return [self.canvas]
        if name == "pqkenviek":
            return list(self.palette)
        if name == "eoqnvkspoa-test":
            return [self.target]
        return []

    def get_sprites(self) -> list[_Sprite]:
        return [self.canvas, self.target, *self.palette]


class _Camera:
    def _calculate_scale_and_offset(self) -> tuple[float, float, float]:
        return 2.0, 10.0, -4.0


class _Env:
    def __init__(self, game: SimpleNamespace) -> None:
        self._game = game
        self.steps: list[tuple[str, dict[str, int] | None]] = []

    def step(self, action: object, data: dict[str, int] | None = None) -> SimpleNamespace:
        self.steps.append((str(action), data))
        if self._game.edjesyzxk:
            self._game.edjesyzxk = False
            return SimpleNamespace(levels_completed=0)
        if "ACTION5" in str(action):
            self._game.edjesyzxk = True
        return SimpleNamespace(levels_completed=0)


def _fake_cd82_game(target: np.ndarray | None = None) -> SimpleNamespace:
    canvas = np.zeros((10, 10), dtype=np.int16)
    if target is None:
        target = apply_cd82_region_fill(canvas, 3, 12)
    palette = [
        _Sprite("pqkenviek", 4, 5, center_color=15),
        _Sprite("pqkenviek", 37, 9, center_color=12),
    ]
    return SimpleNamespace(
        current_level=_Level(canvas, np.asarray(target, dtype=np.int16), palette),
        camera=_Camera(),
        xwmfgtlso=0,
        knqmgavuh=15,
        edjesyzxk=False,
        yfobpcuef=False,
    )


def test_req_arc_wmte_4496_spec_declares_adapter_deepen_artifact() -> None:
    """REQ-ARC-WMTE-4496: OpenSpec names the adapter deepening artifact and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in ("REQ-ARC-WMTE-4496", "SCENARIO-ARC-WMTE-4495"):
        assert ref in spec
    assert exp4494.RESULT_RELATIVE_PATH in spec
    for phrase in (
        "cd82",
        "adapter_registered",
        "solution_labels",
        "offline_reproduced=true",
        "reproduced_levels >= 1",
        "beyond the prior L1",
    ):
        assert phrase in spec
    for field, principle in exp4494.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4496_cd82_derives_palette_action_and_goal_from_env() -> None:
    """REQ-ARC-WMTE-4496: cd82 actions and win distance come from current env sprites."""

    game = _fake_cd82_game()
    env = _Env(game)

    labels = cd82.cd82_action_labels(env, None, ())
    palette_pick = json.loads(labels[0])
    assert palette_pick["action"] == 6
    assert palette_pick["color"] == 12
    assert palette_pick["x"] == pytest.approx((37 + 2) * 2 + 10)
    assert palette_pick["y"] == pytest.approx((9 + 2) * 2 - 4)

    game.knqmgavuh = 12
    move = json.loads(cd82.cd82_action_labels(env, None, ())[0])
    assert move == {"action": 4}

    game.xwmfgtlso = 3
    assert json.loads(cd82.cd82_action_labels(env, None, ())[0]) == {"action": 5}
    assert cd82.cd82_hand_verifier(game, None) > 0.0

    game.current_level.canvas.pixels = np.array(game.current_level.target.pixels, copy=True)
    assert cd82.cd82_hand_verifier(game, None) == 0.0

    key_before = cd82.cd82_state_key(game, None)
    game.knqmgavuh = 15
    assert cd82.cd82_state_key(game, None) != key_before


def test_scenario_arc_wmte_4495_cd82_greedy_l2_fill_delta_is_reverse_engineered() -> None:
    """SCENARIO-ARC-WMTE-4495: cd82 L2 needs a derived multi-fill RE delta."""

    canvas = np.zeros((10, 10), dtype=np.int16)
    target = apply_cd82_region_fill(canvas, 3, 12)
    target = apply_cd82_region_fill(target, 0, 15)
    target = apply_cd82_region_fill(target, 3, 12)
    game = _fake_cd82_game(target)

    assert cd82.cd82_remaining_fills(game) == [(3, 12), (0, 15), (3, 12)]


def test_req_arc_wmte_4496_cd82_adapter_is_registered_and_settles_animation() -> None:
    """REQ-ARC-WMTE-4496: arc_game_adapters exposes the cd82 GameAdapter."""

    adapter = get_adapter("cd82")
    game = _fake_cd82_game()
    env = _Env(game)

    assert "cd82" in adaptered_games()
    assert adapter is not None
    assert adapter.game == "cd82"
    assert adapter.branch_mode == "fresh_env"
    assert adapter.action_labels(env, None, ())
    adapter.apply(env, json.dumps({"action": 5}), None)
    assert len(env.steps) == 2
    assert game.edjesyzxk is False


def test_req_arc_wmte_4496_success_artifact_gate_and_schema_errors(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4496: success requires a new level beyond L1."""

    artifact = exp4494.build_artifact(
        preconditions_checked=_preconditions(),
        target_game="cd82",
        adapter_registered=True,
        solution_labels=[json.dumps({"action": 5})],
        solve_reached_level=2,
        reproduction_gate={"game": "cd82", "reached_level": 2, "reproduced": True},
        depth_cap=80,
        states_expanded=12,
        tests_pass=True,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["schema_errors"] == []
    assert exp4494.artifact_schema_errors(artifact) == []

    out = exp4494.write_artifact(artifact, root=tmp_path)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    mutations = [
        (lambda item: item.pop("honest_verdict"), "missing required"),
        (lambda item: item.__setitem__("honest_verdict", "validated"), "terminal prefix"),
        (
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
            "inference_substrate",
        ),
        (lambda item: item.__setitem__("preconditions_checked", []), "preconditions_checked"),
        (lambda item: item.__setitem__("field_principles", {}), "field_principles"),
        (lambda item: item.__setitem__("adapter_registered", False), "success artifact"),
        (lambda item: item.__setitem__("offline_reproduced", False), "success artifact"),
        (lambda item: item.__setitem__("reproduced_levels", 0), "success artifact"),
    ]

    for mutate, expected in mutations:
        changed = dict(artifact)
        mutate(changed)
        assert any(expected in error for error in exp4494.artifact_schema_errors(changed))

    fabricated = dict(artifact)
    fabricated["offline_reproduced"] = False
    with pytest.raises(ValueError, match="success artifact"):
        exp4494.write_artifact(fabricated, root=tmp_path)


def test_scenario_arc_wmte_4495_runner_writes_injected_l2_success(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4495: injected solver/replay success writes stable JSON."""

    fake_adapter = SimpleNamespace(
        game="cd82",
        apply=lambda env, label, frame: frame,
        warmup_label=None,
        depth_caps={2: 80},
        branch_mode="fresh_env",
    )
    calls: list[tuple[str, int, int]] = []

    def fake_solver_runner(game: str, adapter: object, target_level: int, depth_cap: int):
        calls.append((game, target_level, depth_cap))
        return [json.dumps({"action": 5})], 2, 7

    def fake_reproduction_runner(
        game: str,
        labels: list[str],
        apply_fn: object,
        *,
        warmup_label: str | None,
        claimed_level: int,
    ) -> dict[str, object]:
        return {"game": game, "reached_level": claimed_level, "reproduced": bool(labels)}

    artifact = exp4494.run_experiment(
        root=tmp_path,
        adapter_lookup=lambda game: fake_adapter,
        solver_runner=fake_solver_runner,
        reproduction_runner=fake_reproduction_runner,
        preconditions_checked=_preconditions(),
        tests_pass=True,
    )
    written = json.loads((tmp_path / exp4494.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert calls == [("cd82", 2, 80)]
    assert artifact == written
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1


def test_req_arc_wmte_4496_runner_rejects_missing_resource_preconditions() -> None:
    """REQ-ARC-WMTE-4496: missing import or torch preconditions block before replay."""

    with pytest.raises(RuntimeError, match="blocked_offline_arcade_import_smoke"):
        exp4494.ensure_preconditions_ready(
            {"offline_arcade_import_smoke": False, "torch_import": True}
        )
    with pytest.raises(RuntimeError, match="blocked_torch_import"):
        exp4494.ensure_preconditions_ready(
            {"offline_arcade_import_smoke": True, "torch_import": False}
        )
