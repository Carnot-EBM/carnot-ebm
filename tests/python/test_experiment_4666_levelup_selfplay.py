"""Tests for REQ-ARC-WMTE-4666 / SCENARIO-ARC-WMTE-4666."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot import experiment_4666_levelup_selfplay as exp4666


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "dc22",
        "reached_level": reached_level,
        "states_expanded": 5283,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (
            exp4666.CHECKPOINT_RELATIVE_PATH if reproduced else None
        ),
        "reproduction_gate": {
            "game": "dc22",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": ["move"] * 93,
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text(level: int = 1, total: int = 58) -> str:
    return "\n".join(
        [
            "games:",
            "- game: dc22",
            "  reproducibility: reproduced",
            f"  levels_reproduced: {level}",
            "  solver: old_solver.py",
            "- game: sb26",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            f"reproducible_total_levels: {total}",
            "",
        ]
    )


def test_req_arc_wmte_4666_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4666: OpenSpec declares the bank/checkpoint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4666.SPEC_REFS:
        assert ref in spec
    assert exp4666.RESULT_RELATIVE_PATH in spec
    assert exp4666.CHECKPOINT_RELATIVE_PATH in spec
    assert "dc22" in spec
    assert "fallback_exception=true" in spec
    for field, principle in exp4666.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_target_selection_records_fallback_exception(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4666: selection keeps the skip and conflict ledger."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")

    target, selection = exp4666.select_target(registry_path=registry)

    assert target == "dc22"
    assert selection["selected"] == "dc22"
    assert selection["fallback_exception"] is True
    assert selection["rotation_conflict"] is True
    assert "dc22" in selection["prohibited_targets"]
    assert {"game": "bp35", "reason": "no_grounded_next_level_adapter"} in selection["skipped"]
    assert {"game": "cn04", "reason": "standing_loop_repeated_prior_L2"} in selection["skipped"]


def test_success_artifact_counts_only_new_reproduced_level() -> None:
    """SCENARIO-ARC-WMTE-4666: success requires reproduced progress and checkpoint."""

    artifact = exp4666.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=58,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["bp35: no grounded next-level adapter"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "dc22", "fallback_exception": True},
    )

    assert artifact["honest_verdict"] == "success: dc22_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["target_game"] == "dc22"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_checkpoint_updated"] is True
    assert artifact["registry_updated"] is True
    assert artifact["reproducible_total_levels_before"] == 58
    assert artifact["reproducible_total_levels_after"] == 59
    assert exp4666.artifact_schema_errors(artifact) == []


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4666: same-depth reproduction is not a bank."""

    artifact = exp4666.build_artifact(
        _loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=58,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=["dc22 standing loop reached L1, not beyond prior L1"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "dc22", "fallback_exception": True},
    )

    assert artifact["honest_verdict"] == "complete: dc22_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["registry_updated"] is False


def test_schema_errors_reject_invalid_principle_artifact() -> None:
    """REQ-ARC-WMTE-4666: artifact validation names required schema breaches."""

    payload = {
        "honest_verdict": "dc22 L2",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "live_agent",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "registry_updated": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4666.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:target_game" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "offline_reproduced_without_new_level" in errors
    assert "offline_reproduced_without_registry_update" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_read_standing_loop_result_loads_cached_loop(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4666: cached standing-loop evidence is read from results."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_dc22.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    monkeypatch.setattr(exp4666, "RESULTS", results)

    result = exp4666.read_standing_loop_result("dc22")

    assert result["game"] == "dc22"
    assert result["_standing_loop_reused"] is True


def test_run_standing_loop_invokes_command_and_reads_result(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4666: command wrapper shells to arc_loop_solve."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_dc22.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        del cwd, check, text, stdout, stderr
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(exp4666, "RESULTS", results)
    monkeypatch.setattr(exp4666.subprocess, "run", fake_run)

    result = exp4666.run_standing_loop("dc22", 2)

    assert calls and calls[0][1:] == [
        "scripts/arc_loop_solve.py",
        "--game",
        "dc22",
        "--target-level",
        "2",
        "--no-hazard-prune",
    ]
    assert result["_standing_loop_stdout"] == "ok"


def test_run_standing_loop_raises_on_failure(monkeypatch) -> None:
    """REQ-ARC-WMTE-4666: failed loop commands are not converted into banks."""

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=2, stdout="boom")

    monkeypatch.setattr(exp4666.subprocess, "run", fake_run)

    try:
        exp4666.run_standing_loop("dc22", 2)
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected RuntimeError")


def test_update_registry_for_success_replaces_dc22_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4666: registry stores the bank and dead-end ledger."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")
    artifact = exp4666.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=58,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["bp35: no grounded next-level adapter"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "dc22", "fallback_exception": True},
    )

    changed = exp4666.update_registry_for_success(artifact, registry_path=registry)
    text = registry.read_text(encoding="utf-8")

    assert changed is True
    assert "levels_reproduced: 2" in text
    assert "latest_exp4666_levelup_selfplay" in text
    assert "reproducible_total_levels: 59" in text
    assert "- game: sb26" in text


def test_main_writes_artifact_and_updates_registry(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4666: CLI path writes the stable deliverable."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4666_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_dc22.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    (models / "arc_verifier_dc22.json").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4666, "REGISTRY", registry)
    monkeypatch.setattr(exp4666, "RESULTS", results)
    monkeypatch.setattr(exp4666, "MODELS", models)
    monkeypatch.setattr(exp4666, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4666, "REGISTRY_RELATIVE_PATH", str(registry))
    monkeypatch.setitem(
        __import__("sys").modules,
        "carnot.agentic.arc_solver_kit",
        SimpleNamespace(offline_arcade=lambda: object()),
    )

    assert exp4666.main([]) == 0
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "success: dc22_L2_offline_reproduced"
    assert artifact["registry_updated"] is True
    assert artifact["schema_errors"] == []
