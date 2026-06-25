"""Tests for REQ-ARC-WMTE-4739 / SCENARIO-ARC-WMTE-4739."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4739_levelup_selfplay as exp4739


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "re86",
        "reached_level": reached_level,
        "moves": 56,
        "states_expanded": 56,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": exp4739.CHECKPOINT_RELATIVE_PATH,
        "reproduction_gate": {
            "game": "re86",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": ["l1", "l2"],
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
        "selected_generic_operators": [{"operator": "sprite_overlay_resize_verifier"}],
    }


def _registry_text(level: int = 2, total: int = 64) -> str:
    return "\n".join(
        [
            "schema_version: 1",
            "games:",
            "- game: re86",
            "  reproducibility: reproduced",
            f"  levels_reproduced: {level}",
            "  dead_ends:",
            "  - re86: reset-only sprite overlay verifier repeats L1; derive L2 after replaying L1",
            "- game: s5i5",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            "  dead_ends:",
            "  - s5i5: chain_carried_marker_L2_geometry_stalled_at_x39_y30",
            "- game: g50t",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            "  dead_ends:",
            "  - g50t: clone_replay_L2_route_reached_distance_12_no_bank",
            "- game: r11l",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            "  dead_ends:",
            "  - r11l: prefix_rooted_graph_search_stalled_at_L1",
            "- game: m0r0",
            "  reproducibility: reproduced",
            "  levels_reproduced: 2",
            "- game: cn04",
            "  reproducibility: reproduced",
            "  levels_reproduced: 2",
            f"reproducible_total_levels: {total}",
            "",
        ]
    )


def test_req_arc_wmte_4739_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4739: OpenSpec declares the rotated self-play contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4739.SPEC_REFS:
        assert ref in spec
    assert exp4739.RESULT_RELATIVE_PATH in spec
    assert "scripts/arc_loop_solve.py --game re86 --target-level 3 --no-hazard-prune" in spec
    for field, principle in exp4739.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_target_selection_prechecks_re86_l3_and_records_stalls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4739-ROTATED-PRECHECK: target L3 is not a duplicate."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")

    target, selection = exp4739.select_target(registry_path=registry)

    assert target == "re86"
    assert selection["target_level"] == 3
    assert selection["registry_level_before"] == 2
    assert selection["registry_precheck_passed"] is True
    assert "re86" in selection["dead_ends_by_game"]
    assert "g50t" in selection["dead_ends_by_game"]
    assert "clone_replay_L2_route_reached_distance_12_no_bank" in " ".join(
        selection["dead_ends_by_game"]["g50t"]
    )


def test_target_selection_rejects_duplicate_re86_l3(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4739: already-banked target levels are rejected."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(level=3, total=65), encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate registry precheck"):
        exp4739.select_target(registry_path=registry)


def test_registry_helpers_cover_missing_and_cross_game_dead_ends(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4739: helper parsing is deterministic for sparse registries."""

    registry = {
        "games": [
            "malformed-row",
            {"game": "carrier", "dead_ends": ["re86: cross-row L3 route stalled"]},
            {
                "game": "loose",
                "dead_ends": [
                    "plain note",
                    {"gap_id": "GAP-4739"},
                    {"filled_summary": "filled branch"},
                ],
            },
        ]
    }

    assert exp4739.sha256_file(tmp_path / "missing.json") is None
    assert exp4739._game_entry(registry, "absent") == {}
    assert exp4739._dead_end_notes(registry["games"][2]) == [
        "plain note",
        "GAP-4739",
        "filled branch",
    ]
    assert exp4739._dead_ends_for_game(registry, "re86") == [
        "re86: cross-row L3 route stalled"
    ]


def test_no_bank_artifact_keeps_total_and_checkpoint() -> None:
    """SCENARIO-ARC-WMTE-4739-NO-BANK-RESIDUAL: L2 replay is not counted as L3."""

    artifact = exp4739.build_artifact(
        _loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=64,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["re86: target L3 reached L2; no bank"],
        preconditions_checked=[
            "arc_solver_kit.offline_arcade()",
            "scripts/arc_loop_solve.py --help",
        ],
        target_selection={"selected": "re86", "registry_precheck_passed": True},
    )

    assert artifact["honest_verdict"] == "complete: re86_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 0
    assert artifact["reproducible_total_levels"] == 64
    assert artifact["verifier_checkpoint"] == exp4739.CHECKPOINT_RELATIVE_PATH
    assert artifact["verifier_is_oracle"] is False
    assert artifact["schema_errors"] == []


def test_success_artifact_would_bank_only_reproduced_new_level() -> None:
    """SCENARIO-ARC-WMTE-4739-REPRODUCTION-GATED-SELFPLAY: success needs L3."""

    artifact = exp4739.build_artifact(
        _loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=64,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=[],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "re86", "registry_precheck_passed": True},
    )

    assert artifact["honest_verdict"] == "success: re86_L3_offline_reproduced"
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["schema_errors"] == []


def test_schema_errors_reject_invalid_artifact() -> None:
    """REQ-ARC-WMTE-4739: artifact validation catches schema breaches."""

    payload = {
        "honest_verdict": "re86 L3",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "bogus",
        "offline_reproduced": False,
        "new_levels_banked": 1,
        "registry_precheck_passed": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4739.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:verifier_checkpoint" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "invalid_solve_provenance" in errors
    assert "bank_without_registry_precheck" in errors
    assert "bank_without_offline_reproduction" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_stale_cached_loop_is_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-4739-REPRODUCTION-GATED-SELFPLAY: stale cache reruns."""

    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        exp4739, "read_standing_loop_result", lambda game: _loop_result(reached_level=1)
    )

    def fake_run(game: str, target_level: int) -> dict[str, object]:
        calls.append((game, target_level))
        return _loop_result(reached_level=2)

    monkeypatch.setattr(exp4739, "run_standing_loop", fake_run)

    result = exp4739.load_or_run_standing_loop("re86", 3, 2)

    assert calls == [("re86", 3)]
    assert result["reached_level"] == 2


def test_cached_loop_dead_end_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-4739-NO-BANK-RESIDUAL: cached outcomes are explicit."""

    monkeypatch.setattr(
        exp4739, "read_standing_loop_result", lambda game: _loop_result(reached_level=3)
    )
    assert (
        exp4739._attempt_dead_end_from_loop("re86", 3)
        == "re86: target L3 reproduced; candidate should be registry-gated before any future count"
    )

    monkeypatch.setattr(
        exp4739, "read_standing_loop_result", lambda game: _loop_result(reached_level=2)
    )
    assert exp4739._attempt_dead_end_from_loop("re86", 3) == (
        "re86: target L3 reached L2; no bank"
    )


def test_run_standing_loop_invokes_re86_l3_command_and_reads_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4739-REPRODUCTION-GATED-SELFPLAY: wrapper runs the loop."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_re86.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        del cwd, check, text, stdout, stderr
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(exp4739, "RESULTS", results)
    monkeypatch.setattr(exp4739.subprocess, "run", fake_run)

    result = exp4739.run_standing_loop("re86", 3)

    assert calls and calls[0][1:] == [
        "scripts/arc_loop_solve.py",
        "--game",
        "re86",
        "--target-level",
        "3",
        "--no-hazard-prune",
    ]
    assert result["_standing_loop_stdout"] == "ok"


def test_main_writes_stable_no_bank_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4739-NO-BANK-RESIDUAL: CLI writes the deliverable."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4739_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_re86.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    (models / "arc_verifier_re86.json").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4739, "REGISTRY", registry)
    monkeypatch.setattr(exp4739, "RESULTS", results)
    monkeypatch.setattr(exp4739, "MODELS", models)
    monkeypatch.setattr(exp4739, "ARTIFACT", artifact_path)
    monkeypatch.setattr(
        exp4739,
        "check_preconditions",
        lambda: ["arc_solver_kit.offline_arcade()", "scripts/arc_loop_solve.py --help"],
    )

    exit_code = exp4739.main([])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert artifact["honest_verdict"] == "complete: re86_delta_identified_no_bank"
    assert artifact["schema_errors"] == []
    assert artifact["dead_ends_recorded"]
    assert "m0r0" in " ".join(artifact["dead_ends_recorded"])
