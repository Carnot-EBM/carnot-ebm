"""Tests for REQ-ARC-WMTE-4714 / SCENARIO-ARC-WMTE-4714."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4714_levelup_selfplay as exp4714
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_game_adapters import (
    BP35_L1_LABELS,
    BP35_L2_TAIL_LABELS,
    get_adapter,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "bp35",
        "reached_level": reached_level,
        "states_expanded": 40,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (
            exp4714.CHECKPOINT_RELATIVE_PATH if reproduced else None
        ),
        "reproduction_gate": {
            "game": "bp35",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(BP35_L1_LABELS + BP35_L2_TAIL_LABELS),
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text(level: int = 1, total: int = 62) -> str:
    return "\n".join(
        [
            "games:",
            "- game: bp35",
            "  reproducibility: reproduced",
            f"  levels_reproduced: {level}",
            "  solver: python/carnot/experiment_4480_solve_bp35_goal_directed.py",
            "  dead_ends:",
            "  - no_grounded_next_level_adapter_for_platformer_delta",
            "- game: re86",
            "  reproducibility: reproduced",
            "  levels_reproduced: 2",
            f"reproducible_total_levels: {total}",
            "",
        ]
    )


def test_req_arc_wmte_4714_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4714: OpenSpec declares the bank/checkpoint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4714.SPEC_REFS:
        assert ref in spec
    assert exp4714.RESULT_RELATIVE_PATH in spec
    assert exp4714.CHECKPOINT_RELATIVE_PATH in spec
    assert "scripts/arc_loop_solve.py --game bp35 --target-level 2 --no-hazard-prune" in spec
    for field, principle in exp4714.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4714_bp35_adapter_reproduces_l2() -> None:
    """SCENARIO-ARC-WMTE-4714: adapter labels pass the offline reproduction gate."""

    adapter = get_adapter("bp35")
    assert adapter is not None

    gate = kit.reproduce(
        "bp35",
        BP35_L1_LABELS + BP35_L2_TAIL_LABELS,
        adapter.apply,
        claimed_level=2,
    )

    assert gate["reproduced"] is True
    assert gate["reached_level"] >= 2


def test_target_selection_records_registry_precheck(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4714-ROTATED-TARGET: BP35 is selected after dead-end precheck."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")

    target, selection = exp4714.select_target(registry_path=registry)

    assert target == "bp35"
    assert selection["selected"] == "bp35"
    assert selection["registry_precheck_passed"] is True
    assert selection["registry_level_before"] == 1
    assert selection["target_level"] == 2
    assert "lf52" in selection["prohibited_targets"]
    assert "no_grounded_next_level_adapter_for_platformer_delta" in selection["dead_ends_seen"]


def test_target_selection_rejects_duplicate_bp35_l2(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4714-NO-DUPLICATE: already-banked levels are rejected."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(level=2, total=63), encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate registry precheck"):
        exp4714.select_target(registry_path=registry)


def test_target_selection_requires_existing_bp35_l1(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4714: a deepen cannot fabricate the L1 precondition."""

    registry = tmp_path / "registry.yaml"
    registry.write_text("games:\n- game: re86\n  levels_reproduced: 2\n", encoding="utf-8")

    assert exp4714._game_entry({"games": []}, "bp35") == {}
    with pytest.raises(RuntimeError, match="requires an existing reproduced L1"):
        exp4714.select_target(registry_path=registry)


def test_dead_end_notes_accept_dict_rows() -> None:
    """SCENARIO-ARC-WMTE-4714-ROTATED-TARGET: registry dict dead-ends are summarized."""

    notes = exp4714._dead_end_notes(
        {
            "dead_ends": [
                {"gap_id": "GAP-4480-BP35-GOAL-DIRECTED-NAVIGATION"},
                {"filled_summary": "goal-directed bp35 navigation reproduced L1 offline"},
            ]
        }
    )

    assert notes == [
        "GAP-4480-BP35-GOAL-DIRECTED-NAVIGATION",
        "goal-directed bp35 navigation reproduced L1 offline",
    ]


def test_sha256_file_returns_none_for_missing_path(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4714: absent checkpoint hashes are explicit nulls."""

    assert exp4714.sha256_file(tmp_path / "missing.json") is None


def test_run_standing_loop_invokes_bp35_command_and_reads_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4714-BANK-AND-CHECKPOINT: wrapper shells to arc_loop_solve."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_bp35.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        del cwd, check, text, stdout, stderr
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(exp4714, "RESULTS", results)
    monkeypatch.setattr(exp4714.subprocess, "run", fake_run)

    result = exp4714.run_standing_loop("bp35", 2)

    assert calls and calls[0][1:] == [
        "scripts/arc_loop_solve.py",
        "--game",
        "bp35",
        "--target-level",
        "2",
        "--no-hazard-prune",
    ]
    assert result["_standing_loop_stdout"] == "ok"


def test_load_or_run_standing_loop_refreshes_unbankable_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4714: stale cached loops do not satisfy the bank gate."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_bp35.json").write_text(
        json.dumps(_loop_result(reached_level=1)), encoding="utf-8"
    )
    monkeypatch.setattr(exp4714, "RESULTS", results)
    monkeypatch.setattr(exp4714, "run_standing_loop", lambda game, target: _loop_result())

    result = exp4714.load_or_run_standing_loop("bp35", 2, 1)

    assert result["reached_level"] == 2


def test_success_artifact_exposes_required_fields() -> None:
    """SCENARIO-ARC-WMTE-4714-BANK-AND-CHECKPOINT: success banks +1 and checkpoints."""

    artifact = exp4714.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=62,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["bp35: prior no-grounded-adapter dead-end retired"],
        preconditions_checked=[
            "arc_solver_kit.offline_arcade()",
            "scripts/arc_loop_solve.py --help",
        ],
        target_selection={"selected": "bp35", "registry_precheck_passed": True},
    )

    assert artifact["honest_verdict"] == "success: bp35_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels"] == 63
    assert artifact["verifier_checkpoint"] == exp4714.CHECKPOINT_RELATIVE_PATH
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["registry_precheck_passed"] is True
    assert exp4714.artifact_schema_errors(artifact) == []


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4714-REGISTRY-GATE: same-depth reproduction is not a bank."""

    artifact = exp4714.build_artifact(
        _loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=62,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=[],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "bp35", "registry_precheck_passed": True},
    )

    assert artifact["honest_verdict"] == "complete: bp35_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["reproducible_total_levels"] == 62
    assert exp4714._dead_end_lines({"dead_ends_recorded": []}) == "  dead_ends: []"
    assert any(
        "not beyond prior" in item
        for item in exp4714._dead_ends_from_selection({"dead_ends_seen": []}, 1, 1)
    )


def test_schema_errors_reject_invalid_principle_artifact() -> None:
    """REQ-ARC-WMTE-4714: artifact validation names required schema breaches."""

    payload = {
        "honest_verdict": "bp35 L2",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "live_agent_self_discovery",
        "offline_reproduced": True,
        "new_levels_banked": 0,
        "registry_precheck_passed": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4714.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:verifier_checkpoint" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "offline_reproduced_without_new_level" in errors
    assert "offline_reproduced_without_registry_precheck" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_update_registry_for_success_replaces_bp35_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4714-REGISTRY-GATE: registry stores the BP35 bank."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")
    artifact = exp4714.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=62,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["bp35: prior no-grounded-adapter dead-end retired"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "bp35", "registry_precheck_passed": True},
    )

    changed = exp4714.update_registry_for_success(artifact, registry_path=registry)
    text = registry.read_text(encoding="utf-8")

    assert changed is True
    assert "levels_reproduced: 2" in text
    assert "latest_exp4714_levelup_selfplay" in text
    assert "reproducible_total_levels: 63" in text
    assert "- game: re86" in text


def test_main_writes_artifact_and_updates_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4714-BANK-AND-CHECKPOINT: CLI writes the deliverable."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4714_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_bp35.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    checkpoint = models / "arc_verifier_bp35.json"
    checkpoint.write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4714, "REGISTRY", registry)
    monkeypatch.setattr(exp4714, "RESULTS", results)
    monkeypatch.setattr(exp4714, "MODELS", models)
    monkeypatch.setattr(exp4714, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4714, "check_preconditions", lambda: ["arc_solver_kit.offline_arcade()"])

    exit_code = exp4714.main([])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert artifact["honest_verdict"] == "success: bp35_L2_offline_reproduced"
    assert artifact["schema_errors"] == []
    assert "latest_exp4714_levelup_selfplay" in registry.read_text(encoding="utf-8")


def test_main_override_game_branch_writes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4714: explicit game override still records precheck context."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4714_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_bp35.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    (models / "arc_verifier_bp35.json").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4714, "REGISTRY", registry)
    monkeypatch.setattr(exp4714, "RESULTS", results)
    monkeypatch.setattr(exp4714, "MODELS", models)
    monkeypatch.setattr(exp4714, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4714, "check_preconditions", lambda: ["arc_solver_kit.offline_arcade()"])

    exit_code = exp4714.main(["--game", "bp35"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert artifact["target_selection"]["override"] is True
    assert artifact["target_selection"]["registry_precheck_passed"] is True
