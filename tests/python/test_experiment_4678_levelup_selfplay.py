"""Tests for REQ-ARC-WMTE-4678 / SCENARIO-ARC-WMTE-4678."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot import experiment_4678_levelup_selfplay as exp4678


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "sb26",
        "reached_level": reached_level,
        "moves": 24,
        "states_expanded": 24,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (
            exp4678.CHECKPOINT_RELATIVE_PATH if reproduced else None
        ),
        "selected_generic_operators": [{"operator": "color_match_slot_sequence_verifier"}],
        "reproduction_gate": {
            "game": "sb26",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(exp4678.SB26_L1_LABELS + exp4678.SB26_L2_TAIL_LABELS),
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text(level: int = 1, total: int = 59) -> str:
    return "\n".join(
        [
            "games:",
            "- game: sb26",
            "  reproducibility: reproduced",
            f"  levels_reproduced: {level}",
            "  solver: old_solver.py",
            "  dead_ends: []",
            "- game: re86",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            f"reproducible_total_levels: {total}",
            "",
        ]
    )


def test_req_arc_wmte_4678_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4678: OpenSpec declares the bank/checkpoint schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4678.SPEC_REFS:
        assert ref in spec
    assert exp4678.RESULT_RELATIVE_PATH in spec
    assert exp4678.CHECKPOINT_RELATIVE_PATH in spec
    assert "sb26" in spec
    assert "reproducible_total_levels` from 59 to 60" in spec
    for field, principle in exp4678.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_sb26_adapter_exposes_l1_seed_and_l2_tail() -> None:
    """SCENARIO-ARC-WMTE-4678-BANK-AND-CHECKPOINT: adaptered loop has the L2 delta."""

    from carnot.agentic import arc_game_adapters

    adapter = arc_game_adapters.get_adapter("sb26")

    assert adapter is not None
    assert adapter.depth_caps[1] == len(exp4678.SB26_L1_LABELS)
    assert adapter.depth_caps[2] == len(exp4678.SB26_L2_TAIL_LABELS)
    assert adapter.level_tails[2] == exp4678.SB26_L2_TAIL_LABELS
    assert adapter.action_labels(SimpleNamespace(), frame=SimpleNamespace(levels_completed=0), path=()) == [
        exp4678.SB26_L1_LABELS[0]
    ]
    assert adapter.action_labels(
        SimpleNamespace(),
        frame=SimpleNamespace(levels_completed=1),
        path=(),
    ) == [exp4678.SB26_L2_TAIL_LABELS[0]]


def test_target_selection_chooses_clean_sb26_and_records_dead_ends(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4678-ROTATED-TARGET: selection skips recent/stalled targets."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")

    target, selection = exp4678.select_target(registry_path=registry)

    assert target == "sb26"
    assert selection["selected"] == "sb26"
    assert selection["rotation_conflict"] is False
    assert "sb26" not in selection["prohibited_targets"]
    assert "dc22" in selection["prohibited_targets"]
    assert {"game": "r11l", "reason": "prefix_rooted_graph_search_stalled_at_L1"} in selection[
        "skipped"
    ]
    assert {"game": "lf52", "reason": "prefix_rooted_graph_search_still_pending_or_stalled"} in selection[
        "skipped"
    ]


def test_success_artifact_counts_only_new_reproduced_level() -> None:
    """SCENARIO-ARC-WMTE-4678-BANK-AND-CHECKPOINT: success requires gate+checkpoint."""

    artifact = exp4678.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=59,
        registry_updated=True,
        checkpoint_before_sha=None,
        checkpoint_after_sha="a" * 64,
        dead_ends_recorded=[
            "r11l: prefix-rooted graph search reached only L1 after 20000 expansions"
        ],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "sb26", "rotation_conflict": False},
    )

    assert artifact["honest_verdict"] == "success: sb26_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["target_game"] == "sb26"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_checkpoint_updated"] is True
    assert artifact["registry_updated"] is True
    assert artifact["reproducible_total_levels_before"] == 59
    assert artifact["reproducible_total_levels_after"] == 60
    assert exp4678.artifact_schema_errors(artifact) == []


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4678-BANK-AND-CHECKPOINT: same-depth replay is not a bank."""

    artifact = exp4678.build_artifact(
        _loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=59,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=["sb26 standing loop reached L1, not beyond prior L1"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "sb26", "rotation_conflict": False},
    )

    assert artifact["honest_verdict"] == "complete: sb26_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["registry_updated"] is False


def test_schema_errors_reject_invalid_principle_artifact() -> None:
    """REQ-ARC-WMTE-4678: schema validation names required breaches."""

    payload = {
        "honest_verdict": "sb26 L2",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "live_agent",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "registry_updated": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4678.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:target_game" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "offline_reproduced_without_new_level" in errors
    assert "offline_reproduced_without_registry_update" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_run_standing_loop_invokes_sb26_command_and_reads_result(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4678-BANK-AND-CHECKPOINT: wrapper shells to arc_loop_solve."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_sb26.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        del cwd, check, text, stdout, stderr
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(exp4678, "RESULTS", results)
    monkeypatch.setattr(exp4678.subprocess, "run", fake_run)

    result = exp4678.run_standing_loop("sb26", 2)

    assert calls and calls[0][1:] == [
        "scripts/arc_loop_solve.py",
        "--game",
        "sb26",
        "--target-level",
        "2",
        "--no-hazard-prune",
    ]
    assert result["_standing_loop_stdout"] == "ok"


def test_update_registry_for_success_replaces_sb26_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4678-REGISTRY-GATE: registry stores the bank."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")
    artifact = exp4678.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=59,
        registry_updated=True,
        checkpoint_before_sha=None,
        checkpoint_after_sha="b" * 64,
        dead_ends_recorded=[
            "r11l: prefix-rooted graph search reached only L1 after 20000 expansions"
        ],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "sb26", "rotation_conflict": False},
    )

    changed = exp4678.update_registry_for_success(artifact, registry_path=registry)
    text = registry.read_text(encoding="utf-8")

    assert changed is True
    assert "levels_reproduced: 2" in text
    assert "latest_exp4678_levelup_selfplay" in text
    assert "learned_verifier_checkpoint: models/arc_verifier_sb26.json" in text
    assert "reproducible_total_levels: 60" in text
    assert "- game: re86" in text


def test_main_writes_artifact_and_updates_registry(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4678-REGISTRY-GATE: CLI path writes the deliverable."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4678_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_sb26.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    (models / "arc_verifier_sb26.json").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4678, "REGISTRY", registry)
    monkeypatch.setattr(exp4678, "RESULTS", results)
    monkeypatch.setattr(exp4678, "MODELS", models)
    monkeypatch.setattr(exp4678, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4678, "REGISTRY_RELATIVE_PATH", str(registry))
    monkeypatch.setattr(exp4678, "check_preconditions", lambda: ["arc_solver_kit.offline_arcade()"])

    assert exp4678.main([]) == 0
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "success: sb26_L2_offline_reproduced"
    assert artifact["registry_updated"] is True
    assert artifact["schema_errors"] == []
