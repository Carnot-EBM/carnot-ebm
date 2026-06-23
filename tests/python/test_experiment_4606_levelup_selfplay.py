"""Tests for REQ-CAPSTONE-4606 / SCENARIO-CAPSTONE-4606."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4606_levelup_selfplay as exp4606
from carnot.agentic import arc_game_adapters


def test_spec_mentions_4606_contract() -> None:
    """SCENARIO-CAPSTONE-4606: the level-up self-play artifact is spec anchored."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text()
    assert "REQ-CAPSTONE-4606" in spec
    assert "SCENARIO-CAPSTONE-4606" in spec
    assert "results/experiment_4606_levelup_selfplay.json" in spec


def test_success_artifact_counts_only_new_reproduced_levels() -> None:
    """SCENARIO-CAPSTONE-4606: success requires an offline-reproduced new bank."""

    loop_result = {
        "game": "dc22",
        "reached_level": 2,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "learned_verifier_checkpoint": "models/arc_verifier_dc22.json",
        "reproduction_gate": {"reproduced": True, "reached_level": 2},
        "solution_labels": ["{\"action\":1}", "{\"action\":2}"],
        "solve_provenance": "development_proxy",
        "states_expanded": 23,
    }

    artifact = exp4606.build_artifact(
        loop_result,
        prior_level=1,
        prior_total_levels=55,
        registry_updated=True,
        checkpoint_before_sha=None,
        checkpoint_after_sha="abc123",
        dead_ends_recorded=[],
        preconditions_checked=["offline_arcade_importable"],
    )

    assert artifact["honest_verdict"] == "success: dc22_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["target_game"] == "dc22"
    assert artifact["verifier_checkpoint_updated"] is True
    assert artifact["registry_updated"] is True
    assert artifact["reproducible_total_levels_before"] == 55
    assert artifact["reproducible_total_levels_after"] == 56
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["field_principles"] == exp4606.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == ["REQ-CAPSTONE-4606", "SCENARIO-CAPSTONE-4606"]


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-CAPSTONE-4606: reproduced old depth is honest progress, not a bank."""

    loop_result = {
        "game": "sk48",
        "status": "needs_per_game_RE",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "learned_verifier_checkpoint": None,
        "reproduction_gate": {"reproduced": False, "reached_level": 1},
        "solution_labels": [],
        "solve_provenance": "development_proxy",
    }

    artifact = exp4606.build_artifact(
        loop_result,
        prior_level=1,
        prior_total_levels=55,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=["sk48 chain-permutation L2 remained ungrounded"],
        preconditions_checked=["offline_arcade_importable"],
    )

    assert artifact["honest_verdict"] == "complete: sk48_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["verifier_checkpoint_updated"] is False
    assert artifact["registry_updated"] is False
    assert artifact["dead_ends_recorded"] == ["sk48 chain-permutation L2 remained ungrounded"]


def test_target_selection_skips_recorded_stalls(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: selection skips stalled/deepened rotation targets."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "games:",
                "- game: sk48",
                "  levels_reproduced: 1",
                "- game: dc22",
                "  levels_reproduced: 1",
                "reproducible_total_levels: 55",
                "notes: sk48 remained routing/search-progress only; no offline L2 bank.",
            ]
        )
    )

    target, selection = exp4606.select_target(("sk48", "dc22"), registry_path=registry)

    assert target == "dc22"
    assert selection["skipped"] == [
        {"game": "sk48", "reason": "registry_records_sk48_chain_permutation_no_bank"}
    ]


def test_registry_helpers_and_file_sha(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: registry depth and checksum inputs are deterministic."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "games:\n- game: dc22\n  levels_reproduced: 1\nreproducible_total_levels: 55\n"
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("banked")

    assert exp4606.registry_level("dc22", registry_path=registry) == 1
    assert exp4606.registry_level("missing", registry_path=registry) == 0
    assert exp4606.registry_total_levels(registry_path=registry) == 55
    assert exp4606.sha256_file(payload) == "f1e53b902d342070d2517c1fdfd5d14e1c615d0754ec92d8a3bbe3d3c4c291f5"
    assert exp4606.sha256_file(tmp_path / "missing") is None

    registry.write_text(
        "games:\n- game: dc22\n  levels_reproduced: not-an-int\nreproducible_total_levels: none\n"
    )
    assert exp4606.registry_level("dc22", registry_path=registry) == 0
    assert exp4606.registry_total_levels(registry_path=registry) == 0


def test_target_selection_reports_all_skip_reasons_and_blocks_empty_pool(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: target selection refuses disallowed rotations."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "games:",
                "- game: m0r0",
                "  levels_reproduced: 2",
                "- game: ka59",
                "  levels_reproduced: 1",
                "- game: fresh",
                "  levels_reproduced: 0",
                "reproducible_total_levels: 55",
            ]
        )
    )

    with pytest.raises(RuntimeError, match="no eligible target"):
        exp4606.select_target(("m0r0", "ka59", "fresh"), registry_path=registry)


def test_update_registry_for_success_is_scoped_to_game_block(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: success persists the bank and checkpoint in registry."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "updated: '2026-06-22'",
                "games:",
                "- game: dc22",
                "  reproducibility: reproduced",
                "  levels_reproduced: 1",
                "  win_condition: 'L1 toggle-navigation predicate: old'",
                "  action_model: Keyboard ACTION1-4 move jfva; ACTION6 click payloads old. Reproduced 20-label L1 plan from Exp4467.",
                "- game: other",
                "  levels_reproduced: 1",
                "reproducible_total_levels: 55",
                "",
            ]
        )
    )
    artifact = {
        "target_game": "dc22",
        "reached_level": 2,
        "reproduced_levels": 1,
        "reproducibility_checksum": "f" * 64,
        "reproducible_total_levels_before": 55,
        "reproducible_total_levels_after": 56,
        "verifier_delta": {"checkpoint_path": "models/arc_verifier_dc22.json"},
    }

    assert exp4606.update_registry_for_success(artifact, registry_path=registry) is True
    text = registry.read_text()
    assert "levels_reproduced: 2" in text
    assert "latest_exp4606_levelup_selfplay" in text
    assert "models/arc_verifier_dc22.json" in text
    assert "reproducible_total_levels: 56" in text
    assert "- game: other\n  levels_reproduced: 1" in text


def test_run_standing_loop_reads_current_result(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: wrapper delegates solve to the standing loop."""

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "arc_loop_solve_dc22.json").write_text('{"game":"dc22","reached_level":2}')
    monkeypatch.setattr(exp4606, "REPO", tmp_path)
    monkeypatch.setattr(exp4606, "RESULTS", result_dir)

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        assert cmd[-4:] == ["--game", "dc22", "--target-level", "2"]
        assert cwd == tmp_path
        assert check is False and text is True
        return SimpleNamespace(returncode=0, stdout="standing loop ok")

    monkeypatch.setattr(exp4606.subprocess, "run", fake_run)

    out = exp4606.run_standing_loop("dc22", 2)
    assert out["game"] == "dc22"
    assert out["_standing_loop_stdout"] == "standing loop ok"


def test_run_standing_loop_raises_on_subprocess_failure(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: loop failures do not fabricate an artifact."""

    monkeypatch.setattr(exp4606, "REPO", tmp_path)

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        return SimpleNamespace(returncode=2, stdout="loop failed")

    monkeypatch.setattr(exp4606.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="loop failed"):
        exp4606.run_standing_loop("dc22", 2)


def test_write_artifact_and_main_are_side_effect_controllable(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: main writes the final success artifact after registry update."""

    written: list[dict] = []
    checkpoint = tmp_path / "arc_verifier_dc22.json"
    checkpoint.write_text("after")
    monkeypatch.setattr(exp4606, "MODELS", tmp_path)
    monkeypatch.setattr(exp4606, "select_target", lambda: ("dc22", {"selected": "dc22"}))
    monkeypatch.setattr(exp4606, "registry_level", lambda game: 1)
    monkeypatch.setattr(exp4606, "registry_total_levels", lambda: 55)
    checkpoint_shas = iter([None, "after-sha"])
    monkeypatch.setattr(exp4606, "sha256_file", lambda path: next(checkpoint_shas))
    monkeypatch.setattr(exp4606, "update_registry_for_success", lambda artifact: True)
    monkeypatch.setattr(exp4606, "_write_artifact", lambda payload: written.append(payload))
    monkeypatch.setattr(
        exp4606,
        "run_standing_loop",
        lambda game, target_level: {
            "game": game,
            "reached_level": target_level,
            "offline_reproduced": True,
            "reproduced_levels": target_level,
            "learned_verifier_checkpoint": "models/arc_verifier_dc22.json",
            "reproduction_gate": {"reproduced": True, "reached_level": target_level},
            "solution_labels": ["{\"action\":1}"],
            "solve_provenance": "development_proxy",
        },
    )
    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(arc_solver_kit, "offline_arcade", lambda: object())

    assert exp4606.main([]) == 0
    assert len(written) == 2
    assert written[-1]["honest_verdict"] == "success: dc22_L2_offline_reproduced"
    assert written[-1]["registry_updated"] is True
    assert written[-1]["reproducible_total_levels_after"] == 56


def test_main_override_records_dead_end_when_no_new_level(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: CLI game override still emits honest no-bank."""

    written: list[dict] = []
    monkeypatch.setattr(exp4606, "MODELS", tmp_path)
    monkeypatch.setattr(exp4606, "registry_level", lambda game: 1)
    monkeypatch.setattr(exp4606, "registry_total_levels", lambda: 55)
    monkeypatch.setattr(exp4606, "sha256_file", lambda path: None)
    monkeypatch.setattr(exp4606, "update_registry_for_success", lambda artifact: False)
    monkeypatch.setattr(exp4606, "_write_artifact", lambda payload: written.append(payload))
    monkeypatch.setattr(
        exp4606,
        "run_standing_loop",
        lambda game, target_level: {
            "game": game,
            "reached_level": 1,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "learned_verifier_checkpoint": None,
            "reproduction_gate": {"reproduced": True, "reached_level": 1},
            "solution_labels": [],
        },
    )
    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(arc_solver_kit, "offline_arcade", lambda: object())

    assert exp4606.main(["--game", "dc22"]) == 0
    assert written[-1]["honest_verdict"] == "complete: dc22_delta_identified_no_bank"
    assert written[-1]["dead_ends_recorded"] == [
        "dc22 standing loop reached L1, not beyond prior L1"
    ]
    assert written[-1]["target_selection"] == {"selected": "dc22", "override": True}


def test_write_artifact_serializes_sorted_json(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4606: result artifact is stable JSON on disk."""

    out = tmp_path / "artifact.json"
    exp4606._write_artifact({"b": 1, "a": 2}, path=out)
    assert out.read_text().splitlines()[1].strip() == '"a": 2,'


class _FakeSprite:
    def __init__(
        self,
        name: str,
        x: int,
        y: int,
        width: int,
        height: int,
        tags: list[str],
        interaction: str = "InteractionMode.TANGIBLE",
    ) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.tags = tags
        self.interaction = interaction


class _FakeLevel:
    grid_size = (64, 48)

    def __init__(self, sprites: list[_FakeSprite]) -> None:
        self._sprites = sprites

    def get_sprites_by_tag(self, tag: str) -> list[_FakeSprite]:
        return [sprite for sprite in self._sprites if tag in sprite.tags]


def test_dc22_adapter_discovers_l2_buttons_and_verifier_gradient() -> None:
    """SCENARIO-CAPSTONE-4606: dc22 L2 exposes current env buttons and goal distance."""

    adapter = arc_game_adapters.get_adapter("dc22")
    player = _FakeSprite("jfva", 6, 22, 2, 2, ["jfva"])
    goal = _FakeSprite("goknoi", 22, 4, 2, 2, ["goknoi"])
    game = SimpleNamespace(
        current_level=_FakeLevel(
            [
                _FakeSprite("buezna-blrmbx", 46, 30, 13, 5, ["b", "buezna", "sys_click"]),
                _FakeSprite(
                    "buezna-inzejt-refgps",
                    46,
                    21,
                    13,
                    5,
                    ["a", "buezna", "sys_click"],
                    interaction="InteractionMode.REMOVED",
                ),
                _FakeSprite("buezna-matkhq", 46, 12, 13, 5, ["c", "buezna", "sys_click"]),
                player,
                goal,
            ]
        ),
        qnnpcoyzd=player,
        hfuqkxulm=goal,
    )

    labels = adapter.action_labels(SimpleNamespace(_game=game), frame=SimpleNamespace(levels_completed=1), path=[])
    click_rows = [json.loads(label) for label in labels if json.loads(label)["action"] == 6]
    assert {row["sprite"] for row in click_rows} == {
        "buezna-blrmbx",
        "buezna-inzejt-refgps",
        "buezna-matkhq",
    }
    assert any(row["x"] == 52 and row["y"] == 40 for row in click_rows)
    assert adapter.hand_verifier(game, frame=SimpleNamespace(levels_completed=1)) == 34.0
