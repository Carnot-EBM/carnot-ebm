"""Tests for REQ-ARC-WMTE-4751 / SCENARIO-ARC-WMTE-4751."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from carnot import experiment_4751_levelup_selfplay as exp4751


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "sk48",
        "reached_level": reached_level,
        "moves": 44,
        "states_expanded": 44,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (
            exp4751.CHECKPOINT_RELATIVE_PATH if reproduced else None
        ),
        "reproduction_gate": {
            "game": "sk48",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": ["1", "4"],
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
        "selected_generic_operators": [{"operator": "per_level_reinduction_operator"}],
    }


def _registry_text(level: int = 1, total: int = 64) -> str:
    return "\n".join(
        [
            "schema_version: 1",
            "games:",
            "- game: sk48",
            "  reproducibility: reproduced",
            f"  levels_reproduced: {level}",
            "  win_condition: first-solve L1 adapter-free (E1); 14-action sequence replays offline.",
            "  solver: results/arc_explore_trajectory_sk48.json",
            "  reproduce: re-gated reproduced=True L1.",
            "  gotchas: []",
            "- game: re86",
            "  reproducibility: reproduced",
            "  levels_reproduced: 2",
            "  dead_ends:",
            "  - re86: target L3 reached L2; no bank",
            f"reproducible_total_levels: {total}",
            "",
        ]
    )


def test_req_arc_wmte_4751_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4751: OpenSpec declares the SK48 bank contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4751.SPEC_REFS:
        assert ref in spec
    assert exp4751.RESULT_RELATIVE_PATH in spec
    assert exp4751.CHECKPOINT_RELATIVE_PATH in spec
    assert "scripts/arc_loop_solve.py --game sk48 --target-level 2 --no-hazard-prune" in spec
    for field, principle in exp4751.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4751_sk48_adapter_reproduces_l2() -> None:
    """SCENARIO-ARC-WMTE-4751-REPRODUCTION-GATED-BANK: adapter labels pass reproduce()."""

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_game_adapters import (
        SK48_L1_LABELS,
        SK48_L2_SOLUTION_LABELS,
        SK48_L2_TAIL_LABELS,
        get_adapter,
    )

    adapter = get_adapter("sk48")
    assert adapter is not None
    assert SK48_L2_SOLUTION_LABELS == SK48_L1_LABELS + SK48_L2_TAIL_LABELS

    gate = kit.reproduce("sk48", SK48_L2_SOLUTION_LABELS, adapter.apply, claimed_level=2)

    assert gate["reproduced"] is True
    assert gate["reached_level"] >= 2


def test_target_selection_prechecks_sk48_l2(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4751-REGISTRY-PRECHECK: SK48 L2 is a new level."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")

    target, selection = exp4751.select_target(registry_path=registry)

    assert target == "sk48"
    assert selection["target_level"] == 2
    assert selection["registry_level_before"] == 1
    assert selection["registry_precheck_passed"] is True
    assert selection["registry_total_before"] == 64


def test_target_selection_rejects_duplicate_sk48_l2(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4751: already-banked SK48 L2 is rejected."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(level=2, total=65), encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate registry precheck"):
        exp4751.select_target(registry_path=registry)


def test_target_selection_requires_existing_sk48_l1(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4751: SK48 L2 deepening starts from a reproduced L1 row."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(level=0, total=63), encoding="utf-8")

    with pytest.raises(RuntimeError, match="existing reproduced L1"):
        exp4751.select_target(registry_path=registry)


def test_qwen_cache_helpers_and_precondition_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4751: GGUF precondition discovery is explicit and deterministic."""

    cache_root = tmp_path / "models--unsloth--Qwen3.5-9B-MTP-GGUF"
    gguf = cache_root / "snapshots" / "abc" / "Qwen3.5-9B-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_text("gguf\n", encoding="utf-8")

    assert exp4751.resolve_qwen_gguf(cache_root=cache_root) == gguf
    assert exp4751.resolve_qwen_gguf(cache_root=tmp_path / "missing") is None
    assert exp4751.sha256_file(tmp_path / "missing.json") is None

    model_specs = exp4751.model_specs_from_preconditions(
        [{"resource": "qwen3.5_9b_mtp_gguf_cached", "path": str(gguf), "available": True}]
    )

    assert model_specs == [
        {
            "model_id": exp4751.QWEN_MODEL_ID,
            "model_name": exp4751.QWEN_MODEL_NAME,
            "role": "cached_precondition_not_invoked",
            "path": str(gguf),
            "invoked": False,
        }
    ]
    assert exp4751.model_specs_from_preconditions([]) == []

    monkeypatch.setattr(exp4751, "resolve_qwen_gguf", lambda: None)
    with pytest.raises(RuntimeError, match="blocked_qwen35_mtp_gguf_not_cached"):
        exp4751.check_preconditions()


def test_registry_helper_edge_shapes_are_deterministic() -> None:
    """REQ-ARC-WMTE-4751: registry helpers tolerate non-canonical rows."""

    registry = {
        "games": [
            "scalar-row",
            {"game": "carrier", "dead_ends": ["sk48: cross-row L2 note"]},
            {
                "game": "loose",
                "dead_ends": [
                    "plain note",
                    {"gap_id": "GAP-4751"},
                    {"filled_summary": "filled note"},
                    {"foo": "bar"},
                    {"foo": "bar", "baz": "qux"},
                ],
            },
        ]
    }

    assert exp4751._game_entry(registry, "missing") == {}
    assert exp4751._dead_end_notes(registry["games"][2]) == [
        "plain note",
        "GAP-4751",
        "filled note",
        "foo: bar",
        "{'foo': 'bar', 'baz': 'qux'}",
    ]
    assert exp4751._dead_ends_for_game(registry, "sk48") == ["sk48: cross-row L2 note"]
    assert exp4751._dead_end_lines({}) == "  dead_ends: []"
    assert "sk48: target L2 reached L1; no bank" in exp4751._dead_ends_from_selection(
        {"dead_ends_seen": ["legacy note"]}, reached_level=1, prior_level=1
    )


def test_success_artifact_exposes_required_fields() -> None:
    """SCENARIO-ARC-WMTE-4751-REGISTRY-GATE: success banks +1 and checkpoints."""

    artifact = exp4751.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=64,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["sk48: registry_prechecked L1 before L2"],
        preconditions_checked=[
            {"resource": "qwen3.5_9b_mtp_gguf_cached", "available": True, "path": "model.gguf"},
            {"resource": "arc_solver_kit.offline_arcade", "available": True},
        ],
        target_selection={"selected": "sk48", "registry_precheck_passed": True},
        model_specs=[{"model_id": exp4751.QWEN_MODEL_ID, "invoked": False}],
    )

    assert artifact["honest_verdict"] == "success: sk48_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["verifier_checkpoint"] == exp4751.CHECKPOINT_RELATIVE_PATH
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["registry_precheck_passed"] is True
    assert exp4751.artifact_schema_errors(artifact) == []


def test_schema_errors_reject_invalid_artifact() -> None:
    """REQ-ARC-WMTE-4751: artifact validation names required schema breaches."""

    payload = {
        "honest_verdict": "sk48 L2",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "outer_loop_re",
        "offline_reproduced": False,
        "new_levels_banked": 1,
        "registry_precheck_passed": False,
        "registry_updated": False,
        "reproducibility_checksum": "bad",
        "model_specs": [],
    }

    errors = exp4751.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:verifier_checkpoint" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "bank_without_registry_precheck" in errors
    assert "bank_without_offline_reproduction" in errors
    assert "bank_without_registry_update" in errors
    assert "missing_model_specs" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_update_registry_for_success_replaces_sk48_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4751-REGISTRY-GATE: registry stores the SK48 L2 bank."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")
    artifact = exp4751.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=64,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["sk48: registry_prechecked L1 before L2"],
        preconditions_checked=[{"resource": "arc_solver_kit.offline_arcade", "available": True}],
        target_selection={"selected": "sk48", "registry_precheck_passed": True},
        model_specs=[{"model_id": exp4751.QWEN_MODEL_ID, "invoked": False}],
    )

    changed = exp4751.update_registry_for_success(artifact, registry_path=registry)
    text = registry.read_text(encoding="utf-8")

    assert changed is True
    assert "levels_reproduced: 2" in text
    assert "latest_exp4751_levelup_selfplay" in text
    assert "reproducible_total_levels: 65" in text
    assert "- game: re86" in text
    assert yaml.safe_load(text)["reproducible_total_levels"] == 65


def test_run_standing_loop_invokes_sk48_command_and_reads_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4751-REPRODUCTION-GATED-BANK: wrapper shells to loop."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_sk48.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        del cwd, check, text, stdout, stderr
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(exp4751, "RESULTS", results)
    monkeypatch.setattr(exp4751.subprocess, "run", fake_run)

    result = exp4751.run_standing_loop("sk48", 2)

    assert calls and calls[0][1:] == [
        "scripts/arc_loop_solve.py",
        "--game",
        "sk48",
        "--target-level",
        "2",
        "--no-hazard-prune",
    ]
    assert result["_standing_loop_stdout"] == "ok"


def test_load_or_run_refreshes_stale_loop_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-4751-REPRODUCTION-GATED-BANK: stale cached loops rerun."""

    calls: list[tuple[str, int]] = []

    def fake_run(game: str, target_level: int) -> dict[str, object]:
        calls.append((game, target_level))
        return _loop_result()

    monkeypatch.setattr(exp4751, "read_standing_loop_result", lambda game: _loop_result(1))
    monkeypatch.setattr(exp4751, "run_standing_loop", fake_run)

    result = exp4751.load_or_run_standing_loop("sk48", 2, prior_level=1)

    assert calls == [("sk48", 2)]
    assert result["reached_level"] == 2


def test_main_writes_artifact_and_updates_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-4751-REGISTRY-GATE: CLI writes the stable deliverable."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4751_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_sk48.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    (models / "arc_verifier_sk48.json").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4751, "REGISTRY", registry)
    monkeypatch.setattr(exp4751, "RESULTS", results)
    monkeypatch.setattr(exp4751, "MODELS", models)
    monkeypatch.setattr(exp4751, "ARTIFACT", artifact_path)
    monkeypatch.setattr(
        exp4751,
        "check_preconditions",
        lambda: [
            {"resource": "qwen3.5_9b_mtp_gguf_cached", "available": True, "path": "model.gguf"},
            {"resource": "arc_solver_kit.offline_arcade", "available": True},
        ],
    )

    exit_code = exp4751.main([])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert artifact["honest_verdict"] == "success: sk48_L2_offline_reproduced"
    assert artifact["schema_errors"] == []
    assert "latest_exp4751_levelup_selfplay" in registry.read_text(encoding="utf-8")
