"""Tests for REQ-ARC-WMTE-4690 / SCENARIO-ARC-WMTE-4690."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot import experiment_4690_levelup_selfplay as exp4690


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "lf52",
        "reached_level": reached_level,
        "moves": len(exp4690.LF52_L1_LABELS) + len(exp4690.LF52_L2_TAIL_LABELS),
        "states_expanded": 42,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (
            exp4690.CHECKPOINT_RELATIVE_PATH if reproduced else None
        ),
        "selected_generic_operators": [{"operator": "verifier_router_candidate_ranking"}],
        "reproduction_gate": {
            "game": "lf52",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(exp4690.LF52_L1_LABELS + exp4690.LF52_L2_TAIL_LABELS),
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text(level: int = 1, total: int = 60) -> str:
    return "\n".join(
        [
            "games:",
            "- game: lf52",
            "  reproducibility: reproduced",
            f"  levels_reproduced: {level}",
            "  solver: old_solver.py",
            "  gotchas: []",
            "- game: re86",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            f"reproducible_total_levels: {total}",
            "",
        ]
    )


def test_req_arc_wmte_4690_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4690: OpenSpec declares the bank/checkpoint schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4690.SPEC_REFS:
        assert ref in spec
    assert exp4690.RESULT_RELATIVE_PATH in spec
    assert exp4690.CHECKPOINT_RELATIVE_PATH in spec
    assert "lf52" in spec
    assert "reproducible_total_levels` from 60 to 61" in spec
    for field, principle in exp4690.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_lf52_adapter_exposes_l1_seed_and_l2_tail() -> None:
    """SCENARIO-ARC-WMTE-4690-BANK-AND-CHECKPOINT: adaptered loop has the L2 delta."""

    from carnot.agentic import arc_game_adapters

    adapter = arc_game_adapters.get_adapter("lf52")

    assert adapter is not None
    assert adapter.depth_caps[1] == len(exp4690.LF52_L1_LABELS)
    assert adapter.depth_caps[2] == len(exp4690.LF52_L2_TAIL_LABELS)
    assert adapter.level_tails[2] == exp4690.LF52_L2_TAIL_LABELS
    assert adapter.action_labels(SimpleNamespace(), frame=SimpleNamespace(levels_completed=0), path=()) == [
        exp4690.LF52_L1_LABELS[0]
    ]
    assert adapter.action_labels(
        SimpleNamespace(),
        frame=SimpleNamespace(levels_completed=1),
        path=(),
    ) == [exp4690.LF52_L2_TAIL_LABELS[0]]


def test_lf52_adapter_state_and_features_track_hidden_delta() -> None:
    """SCENARIO-ARC-WMTE-4690-BANK-AND-CHECKPOINT: verifier features are oracle-distinct."""

    from carnot.agentic import arc_game_adapters

    peg = SimpleNamespace(name="fozwvlovdui", grid_x=6, grid_y=7, cdpcbbnfdp=(36, 42))
    carrier = SimpleNamespace(name="hupkpseyuim2", grid_x=4, grid_y=4, cdpcbbnfdp=(24, 24))

    class Grid:
        cdpcbbnfdp = (6, 8)

        def ndtvadsrqf(self, name):
            return [peg] if name == "fozwvlovdui" else []

        def whdmasyorl(self, name):
            return [carrier] if name == "hupkpseyuim2" else []

    core = SimpleNamespace(
        hncnfaqaddg=Grid(),
        wpwvsglgmb=SimpleNamespace(qoifrofmiu=peg),
        whtqurkphir=2,
        asqvqzpfdi=7,
        zvcnglshzcx=False,
        yxhdgwykzi=False,
        iajuzrgttrv=False,
        evxflhofing=False,
    )
    game = SimpleNamespace(ikhhdzfmarl=core)
    adapter = arc_game_adapters.get_adapter("lf52")

    assert adapter is not None
    key = adapter.state_key(game, frame=SimpleNamespace(levels_completed=1))
    features = adapter.featurize(game)

    assert key[0] == 1
    assert key[2] == (("fozwvlovdui", 6, 7, 36, 42),)
    assert features == [1.0, 7.0, 1.0, 1.0, 1.0, 6.0, 8.0]


def test_target_selection_chooses_clean_lf52_and_records_skips(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4690-ROTATED-TARGET: selection skips recent/stalled targets."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")

    target, selection = exp4690.select_target(registry_path=registry)

    assert target == "lf52"
    assert selection["selected"] == "lf52"
    assert selection["rotation_conflict"] is False
    assert "lf52" not in selection["prohibited_targets"]
    assert "sb26" in selection["prohibited_targets"]
    assert {"game": "r11l", "reason": "prefix_rooted_graph_search_stalled_at_L1"} in selection[
        "skipped"
    ]
    assert {"game": "g50t", "reason": "target_offset_L2_delta_not_adaptered_this_run"} in selection[
        "skipped"
    ]


def test_success_artifact_counts_only_new_reproduced_level() -> None:
    """SCENARIO-ARC-WMTE-4690-BANK-AND-CHECKPOINT: success requires gate+checkpoint."""

    artifact = exp4690.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=60,
        registry_updated=True,
        checkpoint_before_sha=None,
        checkpoint_after_sha="a" * 64,
        dead_ends_recorded=[
            "r11l: prefix-rooted graph search reached only L1 after 20000 expansions"
        ],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "lf52", "rotation_conflict": False},
    )

    assert artifact["honest_verdict"] == "success: lf52_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["target_game"] == "lf52"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_checkpoint_updated"] is True
    assert artifact["registry_updated"] is True
    assert artifact["reproducible_total_levels_before"] == 60
    assert artifact["reproducible_total_levels_after"] == 61
    assert exp4690.artifact_schema_errors(artifact) == []


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4690-BANK-AND-CHECKPOINT: same-depth replay is not a bank."""

    artifact = exp4690.build_artifact(
        _loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=60,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=["lf52 standing loop reached L1, not beyond prior L1"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "lf52", "rotation_conflict": False},
    )

    assert artifact["honest_verdict"] == "complete: lf52_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["registry_updated"] is False


def test_schema_errors_reject_invalid_principle_artifact() -> None:
    """REQ-ARC-WMTE-4690: schema validation names required breaches."""

    payload = {
        "honest_verdict": "lf52 L2",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "live_agent",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "registry_updated": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4690.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:target_game" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "offline_reproduced_without_new_level" in errors
    assert "offline_reproduced_without_registry_update" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_run_standing_loop_invokes_lf52_command_and_reads_result(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4690-BANK-AND-CHECKPOINT: wrapper shells to arc_loop_solve."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "arc_loop_solve_lf52.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        del cwd, check, text, stdout, stderr
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(exp4690, "RESULTS", results)
    monkeypatch.setattr(exp4690.subprocess, "run", fake_run)

    result = exp4690.run_standing_loop("lf52", 2)

    assert calls and calls[0][1:] == [
        "scripts/arc_loop_solve.py",
        "--game",
        "lf52",
        "--target-level",
        "2",
        "--no-hazard-prune",
    ]
    assert result["_standing_loop_stdout"] == "ok"


def test_update_registry_for_success_replaces_lf52_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4690-REGISTRY-GATE: registry stores the bank."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(_registry_text(), encoding="utf-8")
    artifact = exp4690.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=60,
        registry_updated=True,
        checkpoint_before_sha=None,
        checkpoint_after_sha="b" * 64,
        dead_ends_recorded=[
            "r11l: prefix-rooted graph search reached only L1 after 20000 expansions"
        ],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "lf52", "rotation_conflict": False},
    )

    changed = exp4690.update_registry_for_success(artifact, registry_path=registry)
    text = registry.read_text(encoding="utf-8")

    assert changed is True
    assert "levels_reproduced: 2" in text
    assert "latest_exp4690_levelup_selfplay" in text
    assert "learned_verifier_checkpoint: models/arc_verifier_lf52.json" in text
    assert "reproducible_total_levels: 61" in text
    assert "- game: re86" in text


def test_main_writes_artifact_and_updates_registry(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4690-REGISTRY-GATE: CLI path writes the deliverable."""

    registry = tmp_path / "registry.yaml"
    results = tmp_path / "results"
    models = tmp_path / "models"
    artifact_path = results / "experiment_4690_levelup_selfplay.json"
    registry.write_text(_registry_text(), encoding="utf-8")
    results.mkdir()
    models.mkdir()
    (results / "arc_loop_solve_lf52.json").write_text(
        json.dumps(_loop_result()), encoding="utf-8"
    )
    (models / "arc_verifier_lf52.json").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(exp4690, "REGISTRY", registry)
    monkeypatch.setattr(exp4690, "RESULTS", results)
    monkeypatch.setattr(exp4690, "MODELS", models)
    monkeypatch.setattr(exp4690, "ARTIFACT", artifact_path)
    monkeypatch.setattr(exp4690, "REGISTRY_RELATIVE_PATH", str(registry))
    monkeypatch.setattr(exp4690, "check_preconditions", lambda: ["arc_solver_kit.offline_arcade()"])

    assert exp4690.main([]) == 0
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "success: lf52_L2_offline_reproduced"
    assert artifact["registry_updated"] is True
    assert artifact["schema_errors"] == []
