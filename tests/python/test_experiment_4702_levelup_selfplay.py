"""Tests for REQ-ARC-WMTE-4702 / SCENARIO-ARC-WMTE-4702."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot import experiment_4702_levelup_selfplay as exp4702
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_game_adapters import (
    RE86_L1_LABELS,
    RE86_L2_TAIL_LABELS,
    get_adapter,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "re86",
        "reached_level": reached_level,
        "states_expanded": 56,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (
            exp4702.CHECKPOINT_RELATIVE_PATH if reproduced else None
        ),
        "reproduction_gate": {
            "game": "re86",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(RE86_L1_LABELS + RE86_L2_TAIL_LABELS),
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text(level: int = 1, total: int = 61) -> str:
    return "\n".join(
        [
            "games:",
            "- game: bp35",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            "- game: re86",
            "  reproducibility: reproduced",
            f"  levels_reproduced: {level}",
            "  solver: python/carnot/experiment_4479_solve_re86.py",
            "  dead_ends:",
            "  - sprite_overlay_L2_delta_not_adaptered_this_run",
            "- game: s5i5",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            f"reproducible_total_levels: {total}",
            "",
        ]
    )


def test_req_arc_wmte_4702_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4702: OpenSpec declares the bank/checkpoint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4702.SPEC_REFS:
        assert ref in spec
    assert exp4702.RESULT_RELATIVE_PATH in spec
    assert exp4702.CHECKPOINT_RELATIVE_PATH in spec
    assert "re86" in spec
    for field, principle in exp4702.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4702_re86_adapter_reproduces_l2() -> None:
    """SCENARIO-ARC-WMTE-4702: adapter labels pass the offline reproduction gate."""

    adapter = get_adapter("re86")
    assert adapter is not None

    gate = kit.reproduce(
        "re86",
        RE86_L1_LABELS + RE86_L2_TAIL_LABELS,
        adapter.apply,
        claimed_level=2,
    )

    assert gate["reproduced"] is True
    assert gate["reached_level"] >= 2


def test_target_selection_records_rotation_skips(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4702: selection keeps the skip ledger."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")

    target, selection = exp4702.select_target(registry_path=registry)

    assert target == "re86"
    assert selection["selected"] == "re86"
    assert selection["fallback_exception"] is False
    assert selection["rotation_conflict"] is False
    assert "lf52" in selection["prohibited_targets"]
    assert {"game": "bp35", "reason": "no_grounded_next_level_adapter"} in selection["skipped"]


def test_target_selection_requires_prior_re86_l1(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4702: a deepen cannot fabricate a missing L1 precondition."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "games:\n- game: bp35\n  levels_reproduced: 1\nreproducible_total_levels: 61\n",
        encoding="utf-8",
    )

    try:
        exp4702.select_target(registry_path=registry)
    except RuntimeError as exc:
        assert "re86 deepen requires" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected RuntimeError")


def test_sha256_file_returns_none_for_missing_path(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4702: absent checkpoint hashes are explicit nulls."""

    assert exp4702.sha256_file(tmp_path / "missing.json") is None


def test_run_standing_loop_invokes_command_and_reads_result(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4702: command wrapper shells to arc_loop_solve."""

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

    monkeypatch.setattr(exp4702, "RESULTS", results)
    monkeypatch.setattr(exp4702.subprocess, "run", fake_run)

    result = exp4702.run_standing_loop("re86", 2)

    assert calls and calls[0][1:] == [
        "scripts/arc_loop_solve.py",
        "--game",
        "re86",
        "--target-level",
        "2",
        "--no-hazard-prune",
    ]
    assert result["_standing_loop_stdout"] == "ok"


def test_run_standing_loop_raises_on_failure(monkeypatch) -> None:
    """REQ-ARC-WMTE-4702: failed loop commands are not converted into banks."""

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=2, stdout="boom")

    monkeypatch.setattr(exp4702.subprocess, "run", fake_run)

    try:
        exp4702.run_standing_loop("re86", 2)
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected RuntimeError")


def test_load_or_run_standing_loop_refreshes_unbankable_cache(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4702: stale cached loops do not satisfy the bank gate."""

    results = tmp_path / "results"
    results.mkdir()
    stale = _loop_result(reached_level=1)
    (results / "arc_loop_solve_re86.json").write_text(json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(exp4702, "RESULTS", results)
    monkeypatch.setattr(exp4702, "run_standing_loop", lambda game, target: _loop_result())

    result = exp4702.load_or_run_standing_loop("re86", 2, 1)

    assert result["reached_level"] == 2


def test_success_artifact_counts_only_new_reproduced_level() -> None:
    """SCENARIO-ARC-WMTE-4702: success requires reproduced progress and checkpoint."""

    artifact = exp4702.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=61,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["bp35: no grounded next-level adapter"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "re86", "fallback_exception": False},
    )

    assert artifact["honest_verdict"] == "success: re86_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["target_game"] == "re86"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_checkpoint_updated"] is True
    assert artifact["registry_updated"] is True
    assert artifact["reproducible_total_levels_before"] == 61
    assert artifact["reproducible_total_levels_after"] == 62
    assert exp4702.artifact_schema_errors(artifact) == []


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4702: same-depth reproduction is not a bank."""

    artifact = exp4702.build_artifact(
        _loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=61,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=[],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "re86", "fallback_exception": False},
    )

    assert artifact["honest_verdict"] == "complete: re86_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["registry_updated"] is False
    assert exp4702._dead_end_lines({"dead_ends_recorded": []}) == "  dead_ends: []"
    assert any(
        "not beyond prior" in item
        for item in exp4702._dead_ends_from_selection({"skipped": []}, 1, 1)
    )


def test_schema_errors_reject_invalid_principle_artifact() -> None:
    """REQ-ARC-WMTE-4702: artifact validation names required schema breaches."""

    payload = {
        "honest_verdict": "re86 L2",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "live_agent",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "registry_updated": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4702.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:target_game" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "offline_reproduced_without_new_level" in errors
    assert "offline_reproduced_without_registry_update" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_schema_errors_reject_rotation_conflict() -> None:
    """REQ-ARC-WMTE-4702: selected target cannot be marked a rotation conflict."""

    artifact = exp4702.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=61,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=[],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "re86", "rotation_conflict": True},
    )

    assert "re86_must_not_be_a_rotation_conflict" in exp4702.artifact_schema_errors(artifact)


def test_update_registry_for_success_replaces_re86_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4702: registry stores the bank and dead-end ledger."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")
    artifact = exp4702.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=61,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["bp35: no grounded next-level adapter"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "re86", "fallback_exception": False},
    )

    changed = exp4702.update_registry_for_success(artifact, registry_path=registry)
    text = registry.read_text(encoding="utf-8")

    assert changed is True
    assert "levels_reproduced: 2" in text
    assert "latest_exp4702_levelup_selfplay" in text
    assert "reproducible_total_levels: 62" in text
    assert "- game: s5i5" in text


def test_main_writes_artifact_and_updates_registry(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4702: CLI path writes the stable deliverable."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4702_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_re86.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    checkpoint = models / "arc_verifier_re86.json"
    checkpoint.write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4702, "REGISTRY", registry)
    monkeypatch.setattr(exp4702, "RESULTS", results)
    monkeypatch.setattr(exp4702, "MODELS", models)
    monkeypatch.setattr(exp4702, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4702, "REGISTRY_RELATIVE_PATH", str(registry))
    monkeypatch.setattr(exp4702, "check_preconditions", lambda: ["arc_solver_kit.offline_arcade()"])

    exit_code = exp4702.main([])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert artifact["honest_verdict"] == "success: re86_L2_offline_reproduced"
    assert artifact["schema_errors"] == []
    assert "latest_exp4702_levelup_selfplay" in registry.read_text(encoding="utf-8")


def test_main_override_game_branch_writes_artifact(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4702: explicit game override still records provenance."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4702_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_re86.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    (models / "arc_verifier_re86.json").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4702, "REGISTRY", registry)
    monkeypatch.setattr(exp4702, "RESULTS", results)
    monkeypatch.setattr(exp4702, "MODELS", models)
    monkeypatch.setattr(exp4702, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4702, "check_preconditions", lambda: ["arc_solver_kit.offline_arcade()"])

    exit_code = exp4702.main(["--game", "re86"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert artifact["target_selection"]["override"] is True
