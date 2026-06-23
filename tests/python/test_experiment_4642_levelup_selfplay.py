"""Tests for REQ-ARC-WMTE-4642 / SCENARIO-ARC-WMTE-4642."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4642_levelup_selfplay as exp4642


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 3, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "ft09",
        "reached_level": reached_level,
        "states_expanded": 25,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": (exp4642.CHECKPOINT_RELATIVE_PATH if reproduced else None),
        "reproduction_gate": {
            "game": "ft09",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(exp4642.FT09_L3_SOLUTION_LABELS),
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def test_req_arc_wmte_4642_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4642: the OpenSpec declares the self-play bank contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4642.SPEC_REFS:
        assert ref in spec
    assert exp4642.RESULT_RELATIVE_PATH in spec
    assert exp4642.CHECKPOINT_RELATIVE_PATH in spec
    assert "ft09" in spec
    assert "fallback exception" in spec
    for field, principle in exp4642.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4642_ft09_adapter_exposes_l3_delta() -> None:
    """REQ-ARC-WMTE-4642: ft09 registers a deterministic L1+L2+L3 adapter."""

    from carnot.agentic import arc_game_adapters as adapters

    adapter = adapters.get_adapter("ft09")
    assert adapter is not None
    assert adapter.featurize is not None
    assert adapter.depth_caps[3] == len(exp4642.FT09_L3_TAIL_LABELS)

    frame2 = SimpleNamespace(levels_completed=2)
    assert adapter.action_labels(
        None,
        frame2,
        (),
    ) == [exp4642.FT09_L3_TAIL_LABELS[0]]
    assert (
        adapter.action_labels(
            None,
            frame2,
            tuple(exp4642.FT09_L3_TAIL_LABELS),
        )
        == []
    )


def test_scenario_arc_wmte_4642_standing_loop_reports_ft09_l3(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4642: the standing loop gates ft09 L3 and checkpoints."""

    code = f"""
import json
from pathlib import Path
import scripts.arc_loop_solve as arc_loop_solve
arc_loop_solve.REPO = Path({str(tmp_path)!r})
arc_loop_solve.CKPT_DIR = Path({str(tmp_path / "models")!r})
arc_loop_solve.CKPT_DIR.mkdir(parents=True, exist_ok=True)
result = arc_loop_solve.solve_adaptered("ft09", 3, hazard_prune=False)
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

    assert result["reached_level"] == 3
    assert result["offline_reproduced"] is True
    assert result["reproduced_levels"] == 3
    assert result["learned_verifier_checkpoint"] == exp4642.CHECKPOINT_RELATIVE_PATH
    assert result["reproduction_gate"]["reproduced"] is True
    assert result["solution_labels"] == list(exp4642.FT09_L3_SOLUTION_LABELS)


def test_success_artifact_counts_only_new_reproduced_level() -> None:
    """SCENARIO-ARC-WMTE-4642: success requires reproduced progress and checkpoint."""

    artifact = exp4642.build_artifact(
        _loop_result(),
        prior_level=2,
        prior_total_levels=56,
        registry_updated=True,
        checkpoint_before_sha="before",
        checkpoint_after_sha="after",
        dead_ends_recorded=["r11l: adapter-free L2 bounded search did not bank"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "ft09", "fallback_exception": True},
    )

    assert artifact["honest_verdict"] == "success: ft09_L3_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_checkpoint_updated"] is True
    assert artifact["registry_updated"] is True
    assert artifact["target_selection"]["fallback_exception"] is True
    assert artifact["reproducible_total_levels_before"] == 56
    assert artifact["reproducible_total_levels_after"] == 57
    assert len(artifact["reproducibility_checksum"]) == 64
    assert exp4642.artifact_schema_errors(artifact) == []


def test_no_new_level_artifact_is_complete_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4642: same-depth reproduction is not a bank."""

    artifact = exp4642.build_artifact(
        _loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=56,
        registry_updated=False,
        checkpoint_before_sha=None,
        checkpoint_after_sha=None,
        dead_ends_recorded=["ft09 standing loop reached L2, not beyond prior L2"],
        preconditions_checked=["arc_solver_kit.offline_arcade()"],
        target_selection={"selected": "ft09", "fallback_exception": True},
    )

    assert artifact["honest_verdict"] == "complete: ft09_delta_identified_no_bank"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["verifier_checkpoint_updated"] is False
    assert artifact["registry_updated"] is False
    assert artifact["dead_ends_recorded"] == ["ft09 standing loop reached L2, not beyond prior L2"]


def test_schema_errors_reject_invalid_principle_artifact() -> None:
    """REQ-ARC-WMTE-4642: artifact validation names every required schema breach."""

    payload = {
        "honest_verdict": "ft09 L3",
        "field_principles": {"honest_verdict": "wrong"},
        "verifier_is_oracle": True,
        "solve_provenance": "live_agent",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "registry_updated": False,
        "reproducibility_checksum": "bad",
    }

    errors = exp4642.artifact_schema_errors(payload)

    assert "missing_principle:honest_verdict" in errors
    assert "missing_field:target_game" in errors
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "solve_provenance_must_be_development_proxy" in errors
    assert "offline_reproduced_without_new_level" in errors
    assert "offline_reproduced_without_registry_update" in errors
    assert "invalid_reproducibility_checksum" in errors


def test_target_selection_records_preferred_skips_and_fallback(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4642: selection keeps the rotation-skip ledger."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "games:",
                "- game: r11l",
                "  levels_reproduced: 1",
                "- game: g50t",
                "  levels_reproduced: 1",
                "- game: bp35",
                "  levels_reproduced: 1",
                "- game: m0r0",
                "  levels_reproduced: 2",
                "- game: cn04",
                "  levels_reproduced: 2",
                "- game: ft09",
                "  levels_reproduced: 2",
                "reproducible_total_levels: 56",
            ]
        ),
        encoding="utf-8",
    )

    target, selection = exp4642.select_target(registry_path=registry)

    assert target == "ft09"
    assert selection["selected"] == "ft09"
    assert selection["fallback_exception"] is True
    assert selection["skipped"] == [
        {"game": "r11l", "reason": "adapter_free_l2_bounded_search_no_bank"},
        {"game": "g50t", "reason": "adapter_free_l2_bounded_search_no_bank"},
        {"game": "bp35", "reason": "adapter_free_l2_bounded_search_timeout"},
        {"game": "m0r0", "reason": "standing_loop_repeated_prior_L2"},
        {"game": "cn04", "reason": "standing_loop_repeated_prior_L2"},
    ]


def test_helpers_and_empty_selection_are_deterministic(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4642: registry helpers and checksums are deterministic."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "games:\n- game: ft09\n  levels_reproduced: 2\nreproducible_total_levels: 56\n",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("banked", encoding="utf-8")

    assert exp4642.registry_level("ft09", registry_path=registry) == 2
    assert exp4642.registry_level("missing", registry_path=registry) == 0
    assert exp4642.registry_total_levels(registry_path=registry) == 56
    assert (
        exp4642.sha256_file(payload)
        == "f1e53b902d342070d2517c1fdfd5d14e1c615d0754ec92d8a3bbe3d3c4c291f5"
    )
    assert exp4642.sha256_file(tmp_path / "missing") is None

    registry.write_text(
        "games:\n- game: ft09\n  levels_reproduced: nope\nreproducible_total_levels: none\n",
        encoding="utf-8",
    )
    assert exp4642.registry_level("ft09", registry_path=registry) == 0
    assert exp4642.registry_total_levels(registry_path=registry) == 0

    with pytest.raises(RuntimeError, match="ft09 fallback"):
        exp4642.select_target(fallback="missing", registry_path=registry)


def test_update_registry_for_success_is_scoped_to_ft09_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4642: registry persistence updates only ft09 and total."""

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "updated: '2026-06-23'",
                "games:",
                "- game: ft09",
                "  reproducibility: reproduced",
                "  levels_reproduced: 2",
                "  win_condition: existing L2.",
                "  solver: results/arc_loop_solve_ft09.json",
                "  gotchas: []",
                "- game: other",
                "  levels_reproduced: 1",
                "reproducible_total_levels: 56",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifact = {
        "target_game": "ft09",
        "reached_level": 3,
        "reproduced_levels": 1,
        "reproducibility_checksum": "e" * 64,
        "reproducible_total_levels_before": 56,
        "reproducible_total_levels_after": 57,
        "verifier_delta": {"checkpoint_path": exp4642.CHECKPOINT_RELATIVE_PATH},
    }

    assert exp4642.update_registry_for_success(artifact, registry_path=registry) is True
    text = registry.read_text(encoding="utf-8")
    assert "levels_reproduced: 3" in text
    assert "latest_exp4642_levelup_selfplay" in text
    assert exp4642.CHECKPOINT_RELATIVE_PATH in text
    assert "reproducible_total_levels: 57" in text
    assert "- game: other\n  levels_reproduced: 1" in text


def test_run_standing_loop_reads_current_result(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4642: the wrapper delegates to the standing loop."""

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "arc_loop_solve_ft09.json").write_text(
        '{"game":"ft09","reached_level":3}',
        encoding="utf-8",
    )
    monkeypatch.setattr(exp4642, "REPO", tmp_path)
    monkeypatch.setattr(exp4642, "RESULTS", result_dir)

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        assert cmd == [
            sys.executable,
            "scripts/arc_loop_solve.py",
            "--game",
            "ft09",
            "--target-level",
            "3",
            "--no-hazard-prune",
        ]
        assert cwd == tmp_path
        assert check is False and text is True
        return SimpleNamespace(returncode=0, stdout="standing loop ok")

    monkeypatch.setattr(exp4642.subprocess, "run", fake_run)

    out = exp4642.run_standing_loop("ft09", 3)
    assert out["game"] == "ft09"
    assert out["_standing_loop_stdout"] == "standing loop ok"


def test_run_standing_loop_raises_on_failure(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4642: loop failures do not fabricate artifacts."""

    monkeypatch.setattr(exp4642, "REPO", tmp_path)

    def fake_run(cmd, cwd, check, text, stdout, stderr):
        return SimpleNamespace(returncode=2, stdout="loop failed")

    monkeypatch.setattr(exp4642.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="loop failed"):
        exp4642.run_standing_loop("ft09", 3)


def test_write_artifact_and_main_are_side_effect_controllable(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4642: main writes final artifact after registry update."""

    written: list[dict[str, object]] = []
    monkeypatch.setattr(
        exp4642,
        "select_target",
        lambda: ("ft09", {"selected": "ft09", "skipped": [], "fallback_exception": True}),
    )
    monkeypatch.setattr(exp4642, "registry_level", lambda game: 2)
    monkeypatch.setattr(exp4642, "registry_total_levels", lambda: 56)
    checkpoint_shas = iter(["before", "after"])
    monkeypatch.setattr(exp4642, "sha256_file", lambda path: next(checkpoint_shas))
    monkeypatch.setattr(exp4642, "update_registry_for_success", lambda artifact: True)
    monkeypatch.setattr(exp4642, "_write_artifact", lambda payload: written.append(payload))
    monkeypatch.setattr(
        exp4642, "run_standing_loop", lambda game, target_level: _loop_result(target_level)
    )
    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(arc_solver_kit, "offline_arcade", lambda: object())

    assert exp4642.main([]) == 0
    assert len(written) == 2
    assert written[-1]["honest_verdict"] == "success: ft09_L3_offline_reproduced"
    assert written[-1]["registry_updated"] is True
    assert written[-1]["reproducible_total_levels_after"] == 57


def test_main_override_records_dead_end_when_no_new_level(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4642: CLI override still emits honest no-bank."""

    written: list[dict[str, object]] = []
    monkeypatch.setattr(exp4642, "registry_level", lambda game: 2)
    monkeypatch.setattr(exp4642, "registry_total_levels", lambda: 56)
    monkeypatch.setattr(exp4642, "sha256_file", lambda path: None)
    monkeypatch.setattr(exp4642, "update_registry_for_success", lambda artifact: False)
    monkeypatch.setattr(exp4642, "_write_artifact", lambda payload: written.append(payload))
    monkeypatch.setattr(exp4642, "run_standing_loop", lambda game, target_level: _loop_result(2))
    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(arc_solver_kit, "offline_arcade", lambda: object())

    assert exp4642.main(["--game", "ft09"]) == 0
    assert written[-1]["honest_verdict"] == "complete: ft09_delta_identified_no_bank"
    assert written[-1]["dead_ends_recorded"] == [
        "ft09 standing loop reached L2, not beyond prior L2"
    ]
    assert written[-1]["target_selection"] == {"selected": "ft09", "override": True}


def test_write_artifact_serializes_sorted_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4642: result artifact is stable JSON."""

    out = tmp_path / "artifact.json"
    exp4642._write_artifact({"b": 1, "a": 2}, path=out)
    assert out.read_text(encoding="utf-8").splitlines()[1].strip() == '"a": 2,'
