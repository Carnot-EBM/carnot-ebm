"""Tests for REQ-ARC-WMTE-4654 / SCENARIO-ARC-WMTE-4654."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4654_levelup_selfplay as exp4654


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "vc33",
        "reached_level": reached_level,
        "states_expanded": 12,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (exp4654.CHECKPOINT_RELATIVE_PATH if reproduced else None),
        "reproduction_gate": {
            "game": "vc33",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(exp4654.VC33_L2_SOLUTION_LABELS),
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def test_req_arc_wmte_4654_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4654: OpenSpec declares the self-play bank contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4654.SPEC_REFS:
        assert ref in spec
    assert exp4654.RESULT_RELATIVE_PATH in spec
    assert exp4654.CHECKPOINT_RELATIVE_PATH in spec
    assert "vc33" in spec
    assert "bp35" in spec and "re86" in spec and "sb26" in spec
    for field, principle in exp4654.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4654_vc33_adapter_exposes_l2_delta() -> None:
    """REQ-ARC-WMTE-4654: vc33 registers the deterministic L1+L2 adapter."""

    from carnot.agentic import arc_game_adapters as adapters

    adapter = adapters.get_adapter("vc33")
    assert adapter is not None
    assert adapter.featurize is not None
    assert adapter.depth_caps[2] == len(exp4654.VC33_L2_TAIL_LABELS)
    assert adapter.level_tails[2] == exp4654.VC33_L2_TAIL_LABELS

    frame0 = SimpleNamespace(levels_completed=0)
    frame1 = SimpleNamespace(levels_completed=1)
    assert adapter.action_labels(None, frame0, ()) == [exp4654.VC33_L1_LABELS[0]]
    assert adapter.action_labels(None, frame1, ()) == [exp4654.VC33_L2_TAIL_LABELS[0]]
    assert adapter.action_labels(None, frame1, tuple(exp4654.VC33_L2_TAIL_LABELS)) == []


def test_scenario_arc_wmte_4654_standing_loop_reports_vc33_l2(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4654: the standing loop gates vc33 L2 and checkpoints."""

    code = f"""
import json
from pathlib import Path
import scripts.arc_loop_solve as arc_loop_solve
arc_loop_solve.REPO = Path({str(tmp_path)!r})
arc_loop_solve.CKPT_DIR = Path({str(tmp_path / "models")!r})
arc_loop_solve.CKPT_DIR.mkdir(parents=True, exist_ok=True)
result = arc_loop_solve.solve_adaptered("vc33", 2, hazard_prune=False)
print("JSON_RESULT=" + json.dumps({{
    "reached_level": result["reached_level"],
    "offline_reproduced": result["offline_reproduced"],
    "reproduced_levels": result["reproduced_levels"],
    "learned_verifier_checkpoint": result["learned_verifier_checkpoint"],
    "reproduction_gate": result["reproduction_gate"],
    "solution_labels": result["solution_labels"],
}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.returncode == 0, proc.stdout
    result_line = next(line for line in proc.stdout.splitlines() if line.startswith("JSON_RESULT="))
    result = json.loads(result_line.removeprefix("JSON_RESULT="))

    assert result["reached_level"] == 2
    assert result["offline_reproduced"] is True
    assert result["reproduced_levels"] == 2
    assert result["learned_verifier_checkpoint"] == exp4654.CHECKPOINT_RELATIVE_PATH
    assert result["reproduction_gate"]["reproduced"] is True
    assert result["solution_labels"] == list(exp4654.VC33_L2_SOLUTION_LABELS)


def test_success_artifact_counts_only_new_reproduced_level() -> None:
    """SCENARIO-ARC-WMTE-4654: success requires reproduced progress and checkpoint."""

    artifact = exp4654.build_artifact(
        _loop_result(),
        prior_level=1,
        prior_total_levels=57,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["bp35: no grounded next-level adapter"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "vc33", "fallback_exception": False},
    )

    assert artifact["honest_verdict"] == "success: vc33_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["target_game"] == "vc33"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_checkpoint_updated"] is True
    assert artifact["registry_updated"] is True
    assert artifact["reproducible_total_levels_before"] == 57
    assert artifact["reproducible_total_levels_after"] == 58
    assert exp4654.artifact_schema_errors(artifact) == []


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4654: same-depth reproduction is not a bank."""

    artifact = exp4654.build_artifact(
        _loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=57,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=["vc33 standing loop reached L1, not beyond prior L1"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "vc33", "fallback_exception": False},
    )

    assert artifact["honest_verdict"] == "complete: vc33_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["registry_updated"] is False


def test_schema_errors_reject_invalid_principle_artifact() -> None:
    """REQ-ARC-WMTE-4654: artifact validation names required schema breaches."""

    payload = {
        "honest_verdict": "vc33 L2",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "live_agent",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "registry_updated": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4654.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:target_game" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "offline_reproduced_without_new_level" in errors
    assert "offline_reproduced_without_registry_update" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_target_selection_records_preferred_skips_and_picks_vc33(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4654: selection keeps the rotation-skip ledger."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "games:",
                "- game: bp35",
                "  levels_reproduced: 1",
                "- game: re86",
                "  levels_reproduced: 1",
                "- game: sb26",
                "  levels_reproduced: 1",
                "- game: vc33",
                "  levels_reproduced: 1",
                "- game: ft09",
                "  levels_reproduced: 3",
                "- game: cn04",
                "  levels_reproduced: 2",
                "reproducible_total_levels: 57",
            ]
        ),
        encoding="utf-8",
    )

    target, selection = exp4654.select_target(registry_path=registry)

    assert target == "vc33"
    assert selection["selected"] == "vc33"
    assert selection["fallback_exception"] is False
    assert selection["skipped"] == [
        {"game": "bp35", "reason": "no_grounded_next_level_adapter"},
        {"game": "re86", "reason": "no_grounded_next_level_adapter"},
        {"game": "sb26", "reason": "no_grounded_next_level_adapter"},
        {"game": "m0r0", "reason": "standing_loop_repeated_prior_L2"},
        {"game": "cn04", "reason": "standing_loop_repeated_prior_L2"},
    ]
    assert "ft09" in selection["prohibited_targets"]


def test_helpers_and_empty_registry_values_are_deterministic(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4654: registry helpers and checksums are deterministic."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "games:\n- game: vc33\n  levels_reproduced: 1\nreproducible_total_levels: 57\n",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("banked", encoding="utf-8")

    assert exp4654.registry_level("vc33", registry_path=registry) == 1
    assert exp4654.registry_level("missing", registry_path=registry) == 0
    assert exp4654.registry_total_levels(registry_path=registry) == 57
    assert (
        exp4654.sha256_file(payload)
        == "f1e53b902d342070d2517c1fdfd5d14e1c615d0754ec92d8a3bbe3d3c4c291f5"
    )
    assert exp4654.sha256_file(tmp_path / "missing") is None

    registry.write_text(
        "games:\n- game: vc33\n  levels_reproduced: nope\nreproducible_total_levels: none\n",
        encoding="utf-8",
    )
    assert exp4654.registry_level("vc33", registry_path=registry) == 0
    assert exp4654.registry_total_levels(registry_path=registry) == 0

    with pytest.raises(RuntimeError, match="vc33 target"):
        exp4654.select_target(registry_path=registry)


def test_update_registry_for_success_is_scoped_to_vc33_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4654: registry persistence updates only vc33 and total."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "updated: '2026-06-24'",
                "games:",
                "- game: vc33",
                "  reproducibility: reproduced",
                "  levels_reproduced: 1",
                "  win_condition: existing L1.",
                "  solver: old",
                "- game: other",
                "  levels_reproduced: 1",
                "reproducible_total_levels: 57",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifact = {
        "target_game": "vc33",
        "reached_level": 2,
        "reproduced_levels": 1,
        "reproducibility_checksum": "e" * 64,
        "reproducible_total_levels_before": 57,
        "reproducible_total_levels_after": 58,
        "verifier_delta": {"checkpoint_path": exp4654.CHECKPOINT_RELATIVE_PATH},
    }

    assert exp4654.update_registry_for_success(artifact, registry_path=registry) is True
    text = registry.read_text(encoding="utf-8")
    assert "levels_reproduced: 2" in text
    assert "latest_exp4654_levelup_selfplay" in text
    assert exp4654.CHECKPOINT_RELATIVE_PATH in text
    assert "reproducible_total_levels: 58" in text
    assert "- game: other\n  levels_reproduced: 1" in text


def test_run_standing_loop_reads_current_result(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4654: the wrapper delegates to the standing loop."""

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "arc_loop_solve_vc33.json").write_text(
        '{"game":"vc33","reached_level":2}',
        encoding="utf-8",
    )
    monkeypatch.setattr(exp4654, "REPO", tmp_path)
    monkeypatch.setattr(exp4654, "RESULTS", result_dir)

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        assert cmd == [
            sys.executable,
            "scripts/arc_loop_solve.py",
            "--game",
            "vc33",
            "--target-level",
            "2",
            "--no-hazard-prune",
        ]
        assert cwd == tmp_path
        assert check is False and text is True
        return SimpleNamespace(returncode=0, stdout="standing loop ok")

    monkeypatch.setattr(exp4654.subprocess, "run", fake_run)

    out = exp4654.run_standing_loop("vc33", 2)
    assert out["game"] == "vc33"
    assert out["_standing_loop_stdout"] == "standing loop ok"


def test_run_standing_loop_raises_on_failure(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4654: loop failures do not fabricate artifacts."""

    monkeypatch.setattr(exp4654, "REPO", tmp_path)

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        return SimpleNamespace(returncode=2, stdout="loop failed")

    monkeypatch.setattr(exp4654.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="loop failed"):
        exp4654.run_standing_loop("vc33", 2)


def test_write_artifact_and_main_are_side_effect_controllable(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4654: main writes final artifact after registry update."""

    written: list[dict[str, object]] = []
    monkeypatch.setattr(
        exp4654,
        "select_target",
        lambda: ("vc33", {"selected": "vc33", "skipped": [], "fallback_exception": False}),
    )
    monkeypatch.setattr(exp4654, "registry_level", lambda game: 1)
    monkeypatch.setattr(exp4654, "registry_total_levels", lambda: 57)
    checkpoint_shas = iter(["before", "after"])
    monkeypatch.setattr(exp4654, "sha256_file", lambda path: next(checkpoint_shas))
    monkeypatch.setattr(exp4654, "update_registry_for_success", lambda artifact: True)
    monkeypatch.setattr(exp4654, "_write_artifact", lambda payload: written.append(payload))
    monkeypatch.setattr(
        exp4654, "run_standing_loop", lambda game, target_level: _loop_result(target_level)
    )
    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(arc_solver_kit, "offline_arcade", lambda: object())

    assert exp4654.main([]) == 0
    assert len(written) == 2
    assert written[-1]["honest_verdict"] == "success: vc33_L2_offline_reproduced"
    assert written[-1]["registry_updated"] is True
    assert written[-1]["reproducible_total_levels_after"] == 58


def test_main_override_records_dead_end_when_no_new_level(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4654: CLI override still emits honest no-bank."""

    written: list[dict[str, object]] = []
    monkeypatch.setattr(exp4654, "registry_level", lambda game: 1)
    monkeypatch.setattr(exp4654, "registry_total_levels", lambda: 57)
    monkeypatch.setattr(exp4654, "sha256_file", lambda path: None)
    monkeypatch.setattr(exp4654, "update_registry_for_success", lambda artifact: False)
    monkeypatch.setattr(exp4654, "_write_artifact", lambda payload: written.append(payload))
    monkeypatch.setattr(exp4654, "run_standing_loop", lambda game, target_level: _loop_result(1))
    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(arc_solver_kit, "offline_arcade", lambda: object())

    assert exp4654.main(["--game", "vc33"]) == 0
    assert written[-1]["honest_verdict"] == "complete: vc33_delta_identified_no_bank"
    assert written[-1]["dead_ends_recorded"] == [
        "vc33 standing loop reached L1, not beyond prior L1"
    ]
    assert written[-1]["target_selection"] == {"selected": "vc33", "override": True}


def test_write_artifact_serializes_sorted_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4654: result artifact is stable JSON."""

    out = tmp_path / "artifact.json"
    exp4654._write_artifact({"b": 1, "a": 2}, path=out)
    assert out.read_text(encoding="utf-8").splitlines()[1].strip() == '"a": 2,'
