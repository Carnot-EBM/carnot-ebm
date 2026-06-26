"""Tests for Exp 4753 .437 lever persistence and transfer characterization.

Spec refs: REQ-ARC-WMTE-4753,
SCENARIO-ARC-WMTE-4753-BLOCKED-PRECONDITION,
SCENARIO-ARC-WMTE-4753-LIVE-PERSISTENCE,
SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4753_persist_transfer as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_artifact(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "honest_verdict": "complete_structured_engine_no_improvement_null",
        "structured_engine_non_degenerate": False,
        "freeform_heldout_accuracy": 0.0,
        "structured_heldout_accuracy": 0.5,
        "accuracy_delta": 0.5,
        "offline_reproduced": False,
        "live_path_reachable": True,
    }
    base.update(overrides)
    return base


def _a2_artifact(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "honest_verdict": "complete_detector_fixed_but_no_bank_no_reachable_plan",
        "detector_pairing_gate": True,
        "detector_goal_count": 2,
        "detector_piece_count": 2,
        "detector_positive_control": {"structural_goal_detected": True},
        "goal_predicate_satisfiable": False,
        "offline_reproduced": False,
        "verifier_is_oracle": False,
    }
    base.update(overrides)
    return base


def _preconditions(ok: bool = True) -> dict[str, Any]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "arcade_import": ok,
        "arcade_import_error": "" if ok else "ModuleNotFoundError: No module named 'arcade'",
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "spec_has_req_4753": True,
        "ok": ok,
        "blocked_resource": "" if ok else "blocked_arcade_import",
    }


def _live_path() -> dict[str, Any]:
    return {
        "agent_path_present": True,
        "structured_engine_env_gate": True,
        "structured_engine_live_import": True,
        "structured_engine_reinduction_proposer": True,
        "structural_alignment_goal_provider": True,
        "structural_goal_passed_to_reinduction": True,
        "persisted_levers": ["structured_engine", "structural_alignment_detector"],
    }


def _lint(passed: bool = True) -> dict[str, Any]:
    return {
        "command": "python scripts/arc_orphan_solver_lint.py",
        "returncode": 0 if passed else 1,
        "passed": passed,
        "stdout_tail": "OK" if passed else "",
        "stderr_tail": "" if passed else "orphan",
    }


def _rows(value: bool = False) -> list[dict[str, Any]]:
    return [
        mod.measure_transfer_game(
            "cn04",
            transfer_row_provider=lambda _game: {
                "baseline_action_efficiency": 0.25,
                "lever_action_efficiency": 0.5 if value else 0.25,
                "baseline_first_effect_step": 4,
                "lever_first_effect_step": 2 if value else 4,
            },
        ),
        mod.measure_transfer_game(
            "r11l",
            transfer_row_provider=lambda _game: {
                "baseline_actions_to_first_effect": 7,
                "lever_actions_to_first_effect": 7,
            },
        ),
    ]


def test_req_arc_wmte_4753_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4753: OpenSpec declares the .437 persist-transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4753",
        "SCENARIO-ARC-WMTE-4753-BLOCKED-PRECONDITION",
        "SCENARIO-ARC-WMTE-4753-LIVE-PERSISTENCE",
        "SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4753_upstream_validation_separates_null_a1_from_fixed_detector() -> None:
    """REQ-ARC-WMTE-4753: upstream gates decide which levers may be persisted."""

    summary = mod.upstream_validation_summary(
        a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )

    assert summary["A1_structured_engine"]["validated"] is False
    assert summary["A1_structured_engine"]["residual"] == "structured_engine_not_validated"
    assert summary["A2_structural_alignment_detector"]["validated"] is True
    assert summary["eligible_levers"] == ["structural_alignment_detector"]

    a1_success = _a1_artifact(
        honest_verdict="success_structured_engine_accuracy_win_lp85",
        structured_engine_non_degenerate=True,
        accuracy_delta=0.5,
    )
    assert mod.upstream_validation_summary(
        a1_artifact=a1_success, a2_artifact=_a2_artifact(detector_pairing_gate=False)
    )["eligible_levers"] == ["structured_engine"]


def test_scenario_arc_wmte_4753_live_path_inspection_finds_real_hooks() -> None:
    """SCENARIO-ARC-WMTE-4753-LIVE-PERSISTENCE: live solver exposes both hooks."""

    persistence = mod.inspect_live_path(REPO)

    assert persistence["structured_engine_env_gate"] is True
    assert persistence["structured_engine_live_import"] is True
    assert persistence["structured_engine_reinduction_proposer"] is True
    assert persistence["structural_alignment_goal_provider"] is True
    assert persistence["structural_goal_passed_to_reinduction"] is True


def test_scenario_arc_wmte_4753_transfer_measurement_reports_efficiency_and_first_effect() -> None:
    """SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION: rows expose transfer deltas."""

    explicit = mod.measure_transfer_game(
        "cn04",
        transfer_row_provider=lambda _game: {
            "action_efficiency_delta": 0.125,
            "first_effect_delta": 3,
            "offline_reproduced_new_level": True,
        },
    )
    assert explicit["value_added"] is True
    assert explicit["transfer_value"]["action_efficiency_delta"] == 0.125
    assert explicit["transfer_value"]["first_effect_delta"] == 3.0
    assert explicit["transfer_value"]["offline_reproduced_new_level"] is False

    computed = mod.measure_transfer_game(
        "r11l",
        transfer_row_provider=lambda _game: {
            "baseline_action_efficiency": 0.2,
            "lever_action_efficiency": 0.45,
            "baseline_first_effect_step": 9,
            "lever_first_effect_step": 5,
        },
    )
    assert computed["transfer_value"]["action_efficiency_delta"] == 0.25
    assert computed["transfer_value"]["first_effect_delta"] == 4.0
    assert computed["value_added"] is True

    count_based = mod.measure_transfer_game(
        "sp80",
        transfer_row_provider=lambda _game: {
            "baseline_actions_to_first_effect": 8,
            "lever_actions_to_first_effect": 6,
        },
    )
    assert count_based["transfer_value"]["first_effect_delta"] == 2.0
    assert count_based["transfer_value"]["action_efficiency_delta"] == 2.0

    provider_error = mod.measure_transfer_game(
        "dc22",
        transfer_row_provider=lambda _game: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    assert provider_error["value_added"] is False
    assert "transfer row unavailable" in provider_error["dead_end"]


def test_scenario_arc_wmte_4753_artifact_schema_for_value_null_and_blocked() -> None:
    """SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION: schema gates all verdicts."""

    validation = mod.upstream_validation_summary(
        a1_artifact=_a1_artifact(structured_engine_non_degenerate=True),
        a2_artifact=_a2_artifact(),
    )
    value = mod.build_artifact(
        upstream_validation=validation,
        live_path_persistence=_live_path(),
        preconditions_checked=_preconditions(True),
        transfer_results=_rows(value=True),
        arc_orphan_solver_lint=_lint(True),
        duration_s=1.25,
    )

    assert value["honest_verdict"] == "complete_437_levers_transfer_value_characterized"
    assert value["offline_reproduced_new_level"] is False
    assert value["transfer_games"] == ["cn04", "r11l"]
    assert mod.artifact_schema_errors(value) == []

    null = mod.build_artifact(
        upstream_validation=validation,
        live_path_persistence=_live_path(),
        preconditions_checked=_preconditions(True),
        transfer_results=_rows(value=False),
        arc_orphan_solver_lint=_lint(True),
        duration_s=1.25,
    )
    assert null["honest_verdict"] == "complete_437_levers_transfer_null_characterized"
    assert "no positive" in null["residual_dead_end"]
    assert mod.artifact_schema_errors(null) == []

    blocked = mod.build_artifact(
        upstream_validation=validation,
        live_path_persistence=_live_path(),
        preconditions_checked=_preconditions(False),
        transfer_results=[],
        arc_orphan_solver_lint={"skipped": "blocked_arcade_import"},
        duration_s=0.01,
    )
    assert blocked["honest_verdict"] == "blocked_arcade_import"
    assert blocked["transfer_games"] == []
    assert blocked["offline_reproduced_new_level"] is False
    assert mod.artifact_schema_errors(blocked) == []


def test_scenario_arc_wmte_4753_run_writes_blocked_and_characterized_artifacts(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4753-BLOCKED-PRECONDITION: run writes requested JSON."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    agent = tmp_path / mod.AGENT_RELATIVE_PATH
    agent.parent.mkdir(parents=True)
    agent.write_text(
        "CARNOT_ARC_STRUCTURED_ENGINE arc_structured_world_model "
        "StructuredEngineReinductionProposer structural_alignment_goal_candidate "
        "structural_goal_provider=structural_goal_provider\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_artifact())

    blocked = mod.run(
        tmp_path,
        arcade_import_checker=lambda: False,
        lint_runner=lambda: _lint(True),
        now=iter([1.0, 1.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_arcade_import"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    characterized = mod.run(
        tmp_path,
        arcade_import_checker=lambda: True,
        transfer_games=("cn04", "r11l"),
        transfer_row_provider=lambda game: {
            "game": game,
            "baseline_action_efficiency": 0.1,
            "lever_action_efficiency": 0.2,
            "baseline_first_effect_step": 5,
            "lever_first_effect_step": 3,
        },
        lint_runner=lambda: _lint(True),
        now=iter([2.0, 2.5]).__next__,
    )

    assert characterized["honest_verdict"] == "complete_437_levers_transfer_value_characterized"
    assert characterized["preconditions_checked"]["ok"] is True
    assert characterized["arc_orphan_solver_lint"]["passed"] is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == characterized


def test_req_arc_wmte_4753_defensive_schema_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4753: malformed inputs remain explicit and checksum-gated."""

    assert "missing required field honest_verdict" in mod.artifact_schema_errors({})
    assert mod._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0

    checks = mod.check_preconditions(
        tmp_path,
        arcade_import_checker=lambda: (_ for _ in ()).throw(RuntimeError("arcade down")),
    )
    assert checks["arcade_import"] is False
    assert checks["blocked_resource"] == "blocked_arcade_import"

    malformed = mod.build_artifact(
        upstream_validation={},
        live_path_persistence={},
        preconditions_checked=_preconditions(True),
        transfer_results=_rows(value=True),
        arc_orphan_solver_lint=_lint(True),
        duration_s=1.0,
    )
    malformed["honest_verdict"] = "bad"
    malformed["inference_substrate"] = "offline"
    malformed["verifier_is_oracle"] = True
    malformed["solve_provenance"] = "outer_loop_re"
    malformed["offline_reproduced_new_level"] = True
    malformed["transfer_games"] = ["one"]
    malformed["transfer_value_per_game"]["one"] = {"offline_reproduced_new_level": True}
    malformed["arc_orphan_solver_lint"] = {"passed": False}
    malformed["field_principles"] = {}
    malformed["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(malformed)
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must be live_llm_inference" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "solve_provenance must be development_proxy" in errors
    assert "offline_reproduced_new_level must be false" in errors
    assert "transfer_games must contain at least two games" in errors
    assert "per-game offline_reproduced_new_level must be false" in errors
    assert "arc_orphan_solver_lint must pass for non-blocked artifacts" in errors
    assert "field_principles must match REQ-ARC-WMTE-4753" in errors
    assert "reproducibility_checksum must match artifact content" in errors

    with pytest.raises(ValueError):
        mod.write_artifact({"honest_verdict": "bad"}, root=tmp_path)
