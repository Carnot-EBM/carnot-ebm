"""Tests for REQ-ARC-WMTE-4728 / SCENARIO-ARC-WMTE-4728."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from carnot import experiment_4728_levelup_selfplay as exp4728


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
AR25_L1_LABELS = tuple(["3"] * 5 + ["2"] * 10)
AR25_L2_TAIL_LABELS = tuple(["3", "3", "5"] + ["2"] * 8)
AR25_L2_SOLUTION_LABELS = AR25_L1_LABELS + AR25_L2_TAIL_LABELS
AR25_L3_TAIL_LABELS = tuple(
    ["1"] * 7 + ["5"] + ["4"] * 7 + ["2"] * 7 + ["5"] + ["3"] * 12 + ["2"] * 5
)
AR25_L3_SOLUTION_LABELS = AR25_L2_SOLUTION_LABELS + AR25_L3_TAIL_LABELS


def _loop_result(reached_level: int = 3, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "ar25",
        "reached_level": reached_level,
        "states_expanded": 88,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (
            exp4728.CHECKPOINT_RELATIVE_PATH if reproduced else None
        ),
        "reproduction_gate": {
            "game": "ar25",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(AR25_L3_SOLUTION_LABELS),
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
        "selected_generic_operators": [{"operator": "per_level_reinduction_operator"}],
    }


def _registry_text(level: int = 2, total: int = 63) -> str:
    return "\n".join(
        [
            "games:",
            "- game: re86",
            "  reproducibility: reproduced",
            "  levels_reproduced: 2",
            "  dead_ends:",
            "  - g50t: clone_replay_L2_route_reached_distance_12_no_bank",
            "  - s5i5: chain_carried_marker_L2_geometry_stalled_at_x39_y30",
            "  - r11l: prefix_rooted_graph_search_stalled_at_L1",
            "- game: g50t",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            "  dead_ends:",
            "  - gap_id: GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT",
            "- game: s5i5",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            "- game: r11l",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            "- game: ar25",
            "  reproducibility: reproduced",
            f"  levels_reproduced: {level}",
            "  dead_ends:",
            "  - ACTION7 undo stack is hidden-state-bound; skip for banked path",
            f"reproducible_total_levels: {total}",
            "",
        ]
    )


def test_req_arc_wmte_4728_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4728: OpenSpec declares the bank/checkpoint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4728.SPEC_REFS:
        assert ref in spec
    assert exp4728.RESULT_RELATIVE_PATH in spec
    assert exp4728.CHECKPOINT_RELATIVE_PATH in spec
    assert "scripts/arc_loop_solve.py --game ar25 --target-level 3 --no-hazard-prune" in spec
    for field, principle in exp4728.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4728_ar25_adapter_reproduces_l3() -> None:
    """SCENARIO-ARC-WMTE-4728-BANK-AND-CHECKPOINT: adapter labels pass reproduce()."""

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_game_adapters import (
        AR25_L2_SOLUTION_LABELS as adapter_l2_labels,
        AR25_L3_SOLUTION_LABELS as adapter_l3_labels,
        AR25_L3_TAIL_LABELS as adapter_l3_tail_labels,
        get_adapter,
    )

    adapter = get_adapter("ar25")
    assert adapter is not None
    assert adapter_l3_labels == AR25_L3_SOLUTION_LABELS
    assert adapter_l3_labels == adapter_l2_labels + adapter_l3_tail_labels

    gate = kit.reproduce("ar25", adapter_l3_labels, adapter.apply, claimed_level=3)

    assert gate["reproduced"] is True
    assert gate["reached_level"] >= 3


def test_target_selection_rotates_to_ar25_after_prechecked_dead_ends(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4728-ROTATED-TARGET: stalled preferred targets are recorded."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")

    target, selection = exp4728.select_target(registry_path=registry)

    assert target == "ar25"
    assert selection["selected"] == "ar25"
    assert selection["registry_precheck_passed"] is True
    assert selection["registry_level_before"] == 2
    assert selection["target_level"] == 3
    assert "g50t" in selection["dead_ends_by_game"]
    assert "clone_replay_L2_route_reached_distance_12_no_bank" in " ".join(
        selection["dead_ends_by_game"]["g50t"]
    )


def test_target_selection_rejects_duplicate_ar25_l3(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4728: already-banked levels are rejected."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(level=3, total=64), encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate registry precheck"):
        exp4728.select_target(registry_path=registry)


def test_target_selection_requires_existing_ar25_l2(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4728: AR25 L3 deepening must start from a banked L2."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(level=1, total=62), encoding="utf-8")

    with pytest.raises(RuntimeError, match="existing reproduced L2"):
        exp4728.select_target(registry_path=registry)


def test_registry_helpers_handle_missing_and_defensive_shapes(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4728: precheck parsing tolerates non-canonical registry rows."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "games:",
                "- scalar-row",
                "- game: weird",
                "  levels_reproduced: nope",
                "  dead_ends:",
                "  - foo: bar",
                "    baz: qux",
                "reproducible_total_levels: nope",
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = exp4728._registry_data(registry)

    assert exp4728.sha256_file(tmp_path / "missing.json") is None
    assert exp4728.registry_level("missing", registry_path=registry) == 0
    assert exp4728.registry_level("weird", registry_path=registry) == 0
    assert exp4728.registry_total_levels(registry_path=registry) == 0
    assert exp4728._dead_ends_for_game(data, "missing") == []
    assert exp4728._dead_end_notes({"dead_ends": [{"foo": "bar", "baz": "qux"}]}) == [
        "{'foo': 'bar', 'baz': 'qux'}"
    ]


def test_success_artifact_exposes_required_fields() -> None:
    """SCENARIO-ARC-WMTE-4728-BANK-AND-CHECKPOINT: success banks +1 and checkpoints."""

    artifact = exp4728.build_artifact(
        _loop_result(),
        prior_level=2,
        prior_total_levels=63,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["g50t: clone_replay_L2_route_reached_distance_12_no_bank"],
        preconditions_checked=[
            "arc_solver_kit.offline_arcade()",
            "scripts/arc_loop_solve.py --help",
        ],
        target_selection={"selected": "ar25", "registry_precheck_passed": True},
    )

    assert artifact["honest_verdict"] == "success: ar25_L3_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels"] == 64
    assert artifact["verifier_checkpoint"] == exp4728.CHECKPOINT_RELATIVE_PATH
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["registry_precheck_passed"] is True
    assert exp4728.artifact_schema_errors(artifact) == []


def test_schema_errors_reject_invalid_artifact() -> None:
    """REQ-ARC-WMTE-4728: artifact validation names required schema breaches."""

    payload = {
        "honest_verdict": "ar25 L3",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "live_agent_self_discovery",
        "offline_reproduced": True,
        "new_levels_banked": 0,
        "registry_precheck_passed": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4728.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:verifier_checkpoint" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "offline_reproduced_without_new_level" in errors
    assert "offline_reproduced_without_registry_precheck" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_update_registry_for_success_replaces_ar25_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4728-REGISTRY-GATE: registry stores the AR25 L3 bank."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")
    artifact = exp4728.build_artifact(
        _loop_result(),
        prior_level=2,
        prior_total_levels=63,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["g50t: clone_replay_L2_route_reached_distance_12_no_bank"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "ar25", "registry_precheck_passed": True},
    )

    changed = exp4728.update_registry_for_success(artifact, registry_path=registry)
    text = registry.read_text(encoding="utf-8")

    assert changed is True
    assert "levels_reproduced: 3" in text
    assert "latest_exp4728_levelup_selfplay" in text
    assert "reproducible_total_levels: 64" in text
    assert "- game: re86" in text
    assert yaml.safe_load(text)["reproducible_total_levels"] == 64


def test_run_standing_loop_invokes_ar25_command_and_reads_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4728-BANK-AND-CHECKPOINT: wrapper shells to arc_loop_solve."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_ar25.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        del cwd, check, text, stdout, stderr
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(exp4728, "RESULTS", results)
    monkeypatch.setattr(exp4728.subprocess, "run", fake_run)

    result = exp4728.run_standing_loop("ar25", 3)

    assert calls and calls[0][1:] == [
        "scripts/arc_loop_solve.py",
        "--game",
        "ar25",
        "--target-level",
        "3",
        "--no-hazard-prune",
    ]
    assert result["_standing_loop_stdout"] == "ok"


def test_load_or_run_refreshes_stale_loop_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-4728-BANK-AND-CHECKPOINT: stale cached loops are rerun."""

    calls: list[tuple[str, int]] = []

    def fake_run(game: str, target_level: int) -> dict[str, object]:
        calls.append((game, target_level))
        return _loop_result()

    monkeypatch.setattr(exp4728, "read_standing_loop_result", lambda game: _loop_result(2))
    monkeypatch.setattr(exp4728, "run_standing_loop", fake_run)

    result = exp4728.load_or_run_standing_loop("ar25", 3, prior_level=2)

    assert calls == [("ar25", 3)]
    assert result["reached_level"] == 3


def test_no_bank_dead_end_and_empty_registry_lines_are_explicit() -> None:
    """REQ-ARC-WMTE-4728: no-bank artifacts record a clear rotation dead-end."""

    dead_ends = exp4728._dead_ends_from_selection(
        {"dead_ends_by_game": {}}, reached_level=2, prior_level=2
    )

    assert exp4728._dead_end_lines({}) == "  dead_ends: []"
    assert "ar25 standing loop reached L2, not beyond prior L2" in dead_ends


def test_main_writes_artifact_and_updates_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4728-REGISTRY-GATE: CLI writes the stable deliverable."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4728_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_ar25.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    (models / "arc_verifier_ar25.json").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4728, "REGISTRY", registry)
    monkeypatch.setattr(exp4728, "RESULTS", results)
    monkeypatch.setattr(exp4728, "MODELS", models)
    monkeypatch.setattr(exp4728, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4728, "check_preconditions", lambda: ["arc_solver_kit.offline_arcade()"])

    exit_code = exp4728.main([])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert artifact["honest_verdict"] == "success: ar25_L3_offline_reproduced"
    assert artifact["schema_errors"] == []
    assert "latest_exp4728_levelup_selfplay" in registry.read_text(encoding="utf-8")


def test_main_override_uses_registry_level_for_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4728-REGISTRY-GATE: CLI override still respects registry state."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4728_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (models / "arc_verifier_ar25.json").write_text("checkpoint\n", encoding="utf-8")
    calls: list[tuple[str, int, int]] = []

    def fake_load(game: str, target_level: int, prior_level: int) -> dict[str, object]:
        calls.append((game, target_level, prior_level))
        return _loop_result()

    monkeypatch.setattr(exp4728, "REGISTRY", registry)
    monkeypatch.setattr(exp4728, "RESULTS", results)
    monkeypatch.setattr(exp4728, "MODELS", models)
    monkeypatch.setattr(exp4728, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4728, "check_preconditions", lambda: ["arc_solver_kit.offline_arcade()"])
    monkeypatch.setattr(exp4728, "load_or_run_standing_loop", fake_load)

    exit_code = exp4728.main(["--game", "ar25"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert calls == [("ar25", 3, 2)]
    assert artifact["target_selection"]["override"] is True
