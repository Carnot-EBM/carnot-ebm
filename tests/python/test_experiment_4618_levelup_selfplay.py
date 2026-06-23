"""Tests for REQ-CAPSTONE-4618 / SCENARIO-CAPSTONE-4618."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4618_levelup_selfplay as exp4618


def test_spec_mentions_4618_contract() -> None:
    """SCENARIO-CAPSTONE-4618: the level-up self-play artifact is spec anchored."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text()
    assert "REQ-CAPSTONE-4618" in spec
    assert "SCENARIO-CAPSTONE-4618" in spec
    assert "results/experiment_4618_levelup_selfplay.json" in spec


def test_success_artifact_counts_only_new_reproduced_levels() -> None:
    """SCENARIO-CAPSTONE-4618: success requires an offline-reproduced new bank."""

    loop_result = {
        "game": "sk48",
        "reached_level": 2,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "learned_verifier_checkpoint": "models/arc_verifier_sk48.json",
        "reproduction_gate": {"reproduced": True, "reached_level": 2},
        "solution_labels": ["{\"action\":1}", "{\"action\":4}"],
        "solve_provenance": "development_proxy",
        "states_expanded": 44,
    }

    artifact = exp4618.build_artifact(
        loop_result,
        prior_level=1,
        prior_total_levels=55,
        registry_updated=True,
        checkpoint_before_sha=None,
        checkpoint_after_sha="abc123",
        dead_ends_recorded=["re86 graph-explore probe exhausted without L2"],
        preconditions_checked=["offline_arcade_importable"],
        target_selection={"selected": "sk48"},
    )

    assert artifact["honest_verdict"] == "success: sk48_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["target_game"] == "sk48"
    assert artifact["verifier_checkpoint_updated"] is True
    assert artifact["registry_updated"] is True
    assert artifact["reproducible_total_levels_before"] == 55
    assert artifact["reproducible_total_levels_after"] == 56
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["field_principles"] == exp4618.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == ["REQ-CAPSTONE-4618", "SCENARIO-CAPSTONE-4618"]
    assert "not dc22/ka59" in artifact["field_principles"]["target_game"]


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-CAPSTONE-4618: reproduced old depth is honest progress, not a bank."""

    artifact = exp4618.build_artifact(
        {
            "game": "wa30",
            "status": "needs_per_game_RE",
            "offline_reproduced": False,
            "learned_verifier_checkpoint": None,
            "reproduction_gate": {"reproduced": False, "reached_level": 1},
            "solution_labels": [],
            "solve_provenance": "development_proxy",
        },
        prior_level=1,
        prior_total_levels=55,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=["wa30 current recipe needs RE"],
        preconditions_checked=["offline_arcade_importable"],
    )

    assert artifact["honest_verdict"] == "complete: wa30_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["verifier_checkpoint_updated"] is False
    assert artifact["registry_updated"] is False
    assert artifact["dead_ends_recorded"] == ["wa30 current recipe needs RE"]


def test_target_selection_rotates_to_sk48_and_skips_stalls(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4618: selection skips recent and stalled rotation targets."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "games:",
                "- game: m0r0",
                "  levels_reproduced: 2",
                "- game: dc22",
                "  levels_reproduced: 1",
                "- game: ka59",
                "  levels_reproduced: 1",
                "- game: cd82",
                "  levels_reproduced: 2",
                "- game: fresh",
                "  levels_reproduced: 0",
                "- game: sk48",
                "  levels_reproduced: 1",
                "notes: sk48 remained routing/search-progress only; no offline L2 bank.",
                "reproducible_total_levels: 55",
            ]
        )
    )

    target, selection = exp4618.select_target(
        ("m0r0", "dc22", "ka59", "cd82", "fresh", "sk48"), registry_path=registry
    )

    assert target == "sk48"
    assert selection["selected"] == "sk48"
    assert selection["skipped"] == [
        {"game": "m0r0", "reason": "deepened_or_failed_in_421_425_rotation_window; not_a_shallow_L1_target"},
        {"game": "dc22", "reason": "deepened_or_failed_in_421_425_rotation_window"},
        {"game": "ka59", "reason": "registry_recorded_hidden_stepcounter_or_stalled_delta"},
        {"game": "cd82", "reason": "no_grounded_L3_delta_recorded; not_a_shallow_L1_target"},
        {"game": "fresh", "reason": "not_a_shallow_L1_target"},
    ]
    assert selection["prior_sk48_dead_end"] == "sk48 remained routing/search-progress only; no offline L2 bank."


def test_target_selection_blocks_empty_pool_and_helpers_are_deterministic(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4618: registry helpers and empty-pool refusal are deterministic."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "games:\n- game: ar25\n  levels_reproduced: 2\nreproducible_total_levels: 55\n"
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("banked")

    assert exp4618.registry_level("ar25", registry_path=registry) == 2
    assert exp4618.registry_level("missing", registry_path=registry) == 0
    assert exp4618.registry_total_levels(registry_path=registry) == 55
    assert exp4618.sha256_file(payload) == "f1e53b902d342070d2517c1fdfd5d14e1c615d0754ec92d8a3bbe3d3c4c291f5"
    assert exp4618.sha256_file(tmp_path / "missing") is None

    registry.write_text(
        "games:\n- game: ar25\n  levels_reproduced: not-an-int\nreproducible_total_levels: none\n"
    )
    assert exp4618.registry_level("ar25", registry_path=registry) == 0
    assert exp4618.registry_total_levels(registry_path=registry) == 0

    with pytest.raises(RuntimeError, match="no eligible target"):
        exp4618.select_target(("ar25",), registry_path=registry)


def test_update_registry_for_success_is_scoped_to_sk48_block(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4618: success persists the sk48 bank and checkpoint."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "updated: '2026-06-22'",
                "games:",
                "- game: sk48",
                "  reproducibility: reproduced",
                "  levels_reproduced: 1",
                "  win_condition: first-solve L1 adapter-free.",
                "  solver: results/arc_explore_trajectory_sk48.json",
                "  reproduce: re-gated reproduced=True L1.",
                "  gotchas: []",
                "- game: other",
                "  levels_reproduced: 1",
                "reproducible_total_levels: 55",
                "",
            ]
        )
    )
    artifact = {
        "target_game": "sk48",
        "reached_level": 2,
        "reproduced_levels": 1,
        "reproducibility_checksum": "f" * 64,
        "reproducible_total_levels_before": 55,
        "reproducible_total_levels_after": 56,
        "verifier_delta": {"checkpoint_path": "models/arc_verifier_sk48.json"},
    }

    assert exp4618.update_registry_for_success(artifact, registry_path=registry) is True
    text = registry.read_text()
    assert "levels_reproduced: 2" in text
    assert "latest_exp4618_levelup_selfplay" in text
    assert "models/arc_verifier_sk48.json" in text
    assert "reproducible_total_levels: 56" in text
    assert "- game: other\n  levels_reproduced: 1" in text


def test_run_standing_loop_reads_current_result(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4618: wrapper delegates solve to the standing loop."""

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "arc_loop_solve_sk48.json").write_text('{"game":"sk48","reached_level":2}')
    monkeypatch.setattr(exp4618, "REPO", tmp_path)
    monkeypatch.setattr(exp4618, "RESULTS", result_dir)

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        assert cmd[-4:] == ["--game", "sk48", "--target-level", "2"]
        assert cwd == tmp_path
        assert check is False and text is True
        return SimpleNamespace(returncode=0, stdout="standing loop ok")

    monkeypatch.setattr(exp4618.subprocess, "run", fake_run)

    out = exp4618.run_standing_loop("sk48", 2)
    assert out["game"] == "sk48"
    assert out["_standing_loop_stdout"] == "standing loop ok"


def test_run_standing_loop_raises_on_subprocess_failure(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4618: loop failures do not fabricate an artifact."""

    monkeypatch.setattr(exp4618, "REPO", tmp_path)

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        return SimpleNamespace(returncode=2, stdout="loop failed")

    monkeypatch.setattr(exp4618.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="loop failed"):
        exp4618.run_standing_loop("sk48", 2)


def test_write_artifact_and_main_are_side_effect_controllable(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4618: main writes the final success artifact after registry update."""

    written: list[dict] = []
    monkeypatch.setattr(exp4618, "MODELS", tmp_path)
    monkeypatch.setattr(
        exp4618,
        "select_target",
        lambda: (
            "sk48",
            {
                "selected": "sk48",
                "prior_sk48_dead_end": exp4618.SK48_PRIOR_DEAD_END,
                "skipped": [{"game": "ka59", "reason": "registry_recorded_hidden_stepcounter_or_stalled_delta"}],
            },
        ),
    )
    monkeypatch.setattr(exp4618, "registry_level", lambda game: 1)
    monkeypatch.setattr(exp4618, "registry_total_levels", lambda: 55)
    checkpoint_shas = iter([None, "after-sha"])
    monkeypatch.setattr(exp4618, "sha256_file", lambda path: next(checkpoint_shas))
    monkeypatch.setattr(exp4618, "update_registry_for_success", lambda artifact: True)
    monkeypatch.setattr(exp4618, "_write_artifact", lambda payload: written.append(payload))
    monkeypatch.setattr(
        exp4618,
        "run_standing_loop",
        lambda game, target_level: {
            "game": game,
            "reached_level": target_level,
            "offline_reproduced": True,
            "reproduced_levels": target_level,
            "learned_verifier_checkpoint": "models/arc_verifier_sk48.json",
            "reproduction_gate": {"reproduced": True, "reached_level": target_level},
            "solution_labels": ["{\"action\":1}"],
            "solve_provenance": "development_proxy",
        },
    )
    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(arc_solver_kit, "offline_arcade", lambda: object())

    assert exp4618.main([]) == 0
    assert len(written) == 2
    assert written[-1]["honest_verdict"] == "success: sk48_L2_offline_reproduced"
    assert written[-1]["registry_updated"] is True
    assert written[-1]["reproducible_total_levels_after"] == 56
    assert written[-1]["dead_ends_recorded"] == [
        "prior sk48 routing-only chain-permutation attempt did not bank L2; this run replaced it with a grounded chain-color reorder adapter.",
        "ka59: registry_recorded_hidden_stepcounter_or_stalled_delta",
    ]


def test_main_override_records_dead_end_when_no_new_level(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4618: CLI game override still emits honest no-bank."""

    written: list[dict] = []
    monkeypatch.setattr(exp4618, "MODELS", tmp_path)
    monkeypatch.setattr(exp4618, "registry_level", lambda game: 1)
    monkeypatch.setattr(exp4618, "registry_total_levels", lambda: 55)
    monkeypatch.setattr(exp4618, "sha256_file", lambda path: None)
    monkeypatch.setattr(exp4618, "update_registry_for_success", lambda artifact: False)
    monkeypatch.setattr(exp4618, "_write_artifact", lambda payload: written.append(payload))
    monkeypatch.setattr(
        exp4618,
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

    assert exp4618.main(["--game", "wa30"]) == 0
    assert written[-1]["honest_verdict"] == "complete: wa30_delta_identified_no_bank"
    assert written[-1]["dead_ends_recorded"] == [
        "wa30 standing loop reached L1, not beyond prior L1"
    ]
    assert written[-1]["target_selection"] == {"selected": "wa30", "override": True}


def test_write_artifact_serializes_sorted_json(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4618: result artifact is stable JSON on disk."""

    out = tmp_path / "artifact.json"
    exp4618._write_artifact({"b": 1, "a": 2}, path=out)
    assert out.read_text().splitlines()[1].strip() == '"a": 2,'


def test_sk48_adapter_exposes_deterministic_l2_tail() -> None:
    """SCENARIO-CAPSTONE-4618: sk48 L2 is registered as a replayable GameAdapter delta."""

    from carnot.agentic import arc_game_adapters

    adapter = arc_game_adapters.get_adapter("sk48")
    assert adapter is not None
    assert adapter.branch_mode == "fresh_env"
    assert adapter.depth_caps[1] == len(arc_game_adapters.SK48_L1_LABELS)
    assert adapter.depth_caps[2] == len(arc_game_adapters.SK48_L2_TAIL_LABELS)
    assert (
        arc_game_adapters.SK48_L2_SOLUTION_LABELS
        == arc_game_adapters.SK48_L1_LABELS + arc_game_adapters.SK48_L2_TAIL_LABELS
    )

    env = SimpleNamespace(_game=SimpleNamespace(level_index=1))
    assert adapter.action_labels(env, frame=SimpleNamespace(levels_completed=0), path=()) == [
        arc_game_adapters.SK48_L1_LABELS[0]
    ]
    assert adapter.action_labels(
        env,
        frame=SimpleNamespace(levels_completed=1),
        path=tuple(arc_game_adapters.SK48_L2_TAIL_LABELS[:3]),
    ) == [arc_game_adapters.SK48_L2_TAIL_LABELS[3]]
    assert adapter.action_labels(
        env,
        frame=SimpleNamespace(levels_completed=2),
        path=tuple(arc_game_adapters.SK48_L2_TAIL_LABELS),
    ) == []
    assert json.loads(arc_game_adapters.SK48_L2_TAIL_LABELS[-1]) == {"action": 4}
