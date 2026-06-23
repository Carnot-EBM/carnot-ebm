"""Tests for Exp 4633 ARC sprint integration gate.

Spec refs: REQ-ARC-WMTE-4633, SCENARIO-ARC-WMTE-4633.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
MODULE_PATH = REPO / "python" / "carnot" / "experiment_4633_integration_gate.py"
SUBMITTED_AGENT_CONFIG = {
    "policy": "E3AgentPolicy",
    "value_weight": 0.0,
    "target_levels": 3,
    "frame_change_predictor_enabled": True,
    "frame_change_ranking_mode": "persistent_aem_plus_optional_cnn",
    "dense_curiosity_progress_loop_enabled": False,
    "dense_curiosity_weight": 0.15,
    "dense_curiosity_discount": 0.5,
    "live_submit_package_path": "results/experiment_4631_submission_package_operator_resubmit.json",
    "live_submit_source": "experiment_4631_refresh_submission_package",
}


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "experiment_4633_integration_gate_under_test", MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _install_submitted_config_stub(monkeypatch: Any) -> None:
    carnot = types.ModuleType("carnot")
    agentic = types.ModuleType("carnot.agentic")
    competition = types.ModuleType("carnot.agentic.arc_competition_agent")
    competition.SUBMITTED_AGENT_CONFIG = dict(SUBMITTED_AGENT_CONFIG)
    monkeypatch.setitem(sys.modules, "carnot", carnot)
    monkeypatch.setitem(sys.modules, "carnot.agentic", agentic)
    monkeypatch.setitem(sys.modules, "carnot.agentic.arc_competition_agent", competition)


def _a1_clean_null() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened",
        "flagged_adversarial": None,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "live_solve_rate_bare": 0.04,
        "live_solve_rate_loop": 0.04,
        "solve_rate_delta": 0.0,
        "state_coverage_delta": 2,
        "first_win_rate_delta": 0.0,
        "chosen_submitted_config": "unchanged",
    }


def _a1_metric_success() -> dict[str, Any]:
    return {
        "honest_verdict": "success: dense_curiosity_loop_live_solverate_up_1",
        "flagged_adversarial": None,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "live_solve_rate_bare": 0.04,
        "live_solve_rate_loop": 0.08,
        "solve_rate_delta": 0.04,
        "state_coverage_delta": 12,
        "first_win_rate_delta": 0.04,
    }


def _a2_clean_success() -> dict[str, Any]:
    return {
        "honest_verdict": "success: action_effect_predictor_graduated_live_efficiency_up_1",
        "flagged_adversarial": None,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "solve_rate_preserved": True,
        "actions_delta": 1.0,
        "first_win_rate_delta": 0.25,
        "efficiency_score_term": 1.0,
        "median_actions_to_first_levelup_bare": 2.0,
        "median_actions_to_first_levelup_predictor": 1.0,
        "aggregate_metrics": {
            "first_win_rate_bare": 0.25,
            "first_win_rate_predictor": 0.5,
            "solve_rate_bare": 1.0,
            "solve_rate_predictor": 1.0,
        },
        "chosen_submitted_config": "frame_change_predictor_enabled:persistent_aem_plus_optional_cnn",
    }


def _a2_control_failed() -> dict[str, Any]:
    payload = _a2_clean_success()
    payload["bare_control_passed"] = False
    return payload


def _a3_clean_bank() -> dict[str, Any]:
    return {
        "honest_verdict": "success: ls20_L2_offline_reproduced",
        "flagged_adversarial": None,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "reproduction_gate": {
            "game": "ls20",
            "claimed_level": 2,
            "reached_level": 2,
            "reproduced": True,
        },
    }


def _a4_clean_package() -> dict[str, Any]:
    return {
        "honest_verdict": "success: package_refreshed_live_submittable_56_above_33",
        "flagged_adversarial": None,
        "live_submittable_level_count": 56,
        "live_submittable_count_prev": 55,
        "count_delta": 1,
        "ready_for_operator_submit": True,
        "refreshed_package_path": "results/experiment_4631_submission_package_operator_resubmit.json",
    }


def _summary_clean() -> dict[str, Any]:
    return {"returncode": 0, "live_status": "clean", "stdout": "LIVE re-check: clean"}


def _summary_critical() -> dict[str, Any]:
    return {"returncode": 2, "live_status": "CRITICAL", "stdout": "LIVE re-check: CRITICAL"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4633_spec_declares_integration_gate_contract() -> None:
    """REQ-ARC-WMTE-4633: OpenSpec declares the 4633 artifact schema."""

    mod = _load_mod()

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4633" in spec
    assert "SCENARIO-ARC-WMTE-4633" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4633_audits_clean_metric_winners_and_quarantines_nulls() -> None:
    """REQ-ARC-WMTE-4633: only clean success/control-passed metric raisers integrate."""

    mod = _load_mod()
    audit = mod.audit_upstream_levers(
        {
            "A1": (_a1_clean_null(), _summary_clean()),
            "A2": (_a2_clean_success(), _summary_clean()),
            "A3": (_a3_clean_bank(), _summary_clean()),
        }
    )

    assert audit["levers_integrated"] == [
        "A2_action_effect_ranker",
        "A3_level_bank_refreshed_package",
    ]
    assert audit["submitted_config_raised_metric_clean"] is True
    assert audit["flagged_artifacts_excluded"] == []
    assert audit["upstream_lever_audit"]["A1"]["reason"] == "honest_verdict_not_success"
    assert audit["upstream_lever_audit"]["A2"]["reason"] == "admitted_clean_metric_raiser"
    assert audit["upstream_lever_audit"]["A3"]["reason"] == "admitted_clean_metric_raiser"

    flagged = mod.audit_upstream_levers(
        {
            "A1": (_a1_metric_success(), _summary_critical()),
            "A2": (_a2_control_failed(), _summary_clean()),
            "A3": (_a3_clean_bank(), _summary_clean()),
        }
    )
    assert flagged["upstream_lever_audit"]["A1"]["reason"] == "flagged_adversarial"
    assert flagged["upstream_lever_audit"]["A2"]["reason"] == "positive_control_failed"
    assert flagged["flagged_artifacts_excluded"] == [
        {
            "lever": "A1",
            "reason": "flagged_adversarial",
            "live_status": "CRITICAL",
            "honest_verdict": _a1_metric_success()["honest_verdict"],
        },
        {
            "lever": "A2",
            "reason": "positive_control_failed",
            "live_status": "clean",
            "honest_verdict": _a2_control_failed()["honest_verdict"],
        },
    ]


def test_scenario_arc_wmte_4633_builds_success_artifact_with_required_metrics() -> None:
    """SCENARIO-ARC-WMTE-4633: A2/A3 winners produce a schema-valid shipped artifact."""

    mod = _load_mod()
    audit = mod.audit_upstream_levers(
        {
            "A1": (_a1_clean_null(), _summary_clean()),
            "A2": (_a2_clean_success(), _summary_clean()),
            "A3": (_a3_clean_bank(), _summary_clean()),
        }
    )
    metrics = mod.measure_integrated_metrics(
        audit=audit,
        a1_artifact=_a1_clean_null(),
        a2_artifact=_a2_clean_success(),
        a4_artifact=_a4_clean_package(),
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        orphan_lint={"passed": True, "command": "arc_orphan_solver_lint"},
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        duration_s=1.0,
    )

    assert (
        artifact["honest_verdict"] == "success: integrated_action_efficiency_raised_config_shipped"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["levers_integrated"] == [
        "A2_action_effect_ranker",
        "A3_level_bank_refreshed_package",
    ]
    assert artifact["live_solve_rate_integrated"] == 0.04
    assert artifact["live_solve_rate_delta_vs_bare"] == 0.0
    assert artifact["action_efficiency_integrated"] == {
        "median_actions_to_first_levelup": 1.0,
        "median_actions_to_first_levelup_bare": 2.0,
        "actions_delta_vs_bare": 1.0,
        "efficiency_score_term": 1.0,
    }
    assert artifact["offline_to_live_transfer_ratio_integrated"] == 2.0
    assert artifact["live_submittable_level_count_integrated"] == 56
    assert artifact["live_submittable_delta_vs_baseline"] == 1
    assert artifact["submitted_agent_config"]["frame_change_predictor_enabled"] is True
    assert (
        artifact["submitted_agent_config"]["frame_change_ranking_mode"]
        == "persistent_aem_plus_optional_cnn"
    )
    assert artifact["submitted_agent_config"]["dense_curiosity_progress_loop_enabled"] is False
    assert artifact["submitted_agent_config"]["live_submit_package_path"] == (
        "results/experiment_4631_submission_package_operator_resubmit.json"
    )
    assert "live_solve_rate_delta_vs_bare" in artifact["null_delta_methodology_note"]
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4633_run_writes_artifact_with_summarize_and_lint_gates(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SCENARIO-ARC-WMTE-4633: run reads A1/A2/A3 via summarize and writes JSON."""

    mod = _load_mod()
    _install_submitted_config_stub(monkeypatch)

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-4633\n", encoding="utf-8")
    payloads = {
        mod.A1_RELATIVE_PATH: _a1_clean_null(),
        mod.A2_RELATIVE_PATH: _a2_clean_success(),
        mod.A3_RELATIVE_PATH: _a3_clean_bank(),
        mod.A4_RELATIVE_PATH: _a4_clean_package(),
    }
    for relative, payload in payloads.items():
        _write_json(tmp_path / relative, payload)

    seen: list[str] = []

    def summarize(path: Path) -> dict[str, Any]:
        seen.append(path.name)
        return _summary_clean()

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        summarize_runner=summarize,
        parity_check=lambda _root: {"passed": True, "command": "pytest parity"},
        orphan_lint=lambda _root: {"passed": True, "command": "arc_orphan_solver_lint"},
        now=lambda: 10.0,
    )

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    assert seen == [
        "experiment_4628_dense_curiosity_progress_loop.json",
        "experiment_4629_graduate_action_effect_predictor_live.json",
        "experiment_4630_levelup_selfplay.json",
    ]
    assert artifact["preconditions_checked"]["a1_artifact_present"] is True
    assert artifact["preconditions_checked"]["a4_artifact_present"] is True
    assert artifact["parity_test_green"] is True
    assert artifact["orphan_lint_green"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4633_honest_null_blocked_and_schema_edges(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """REQ-ARC-WMTE-4633: null and blocked paths fail closed without fabricated metrics."""

    mod = _load_mod()
    _install_submitted_config_stub(monkeypatch)

    audit = mod.audit_upstream_levers(
        {
            "A1": (_a1_clean_null(), _summary_clean()),
            "A2": (
                {
                    **_a2_clean_success(),
                    "honest_verdict": "complete: action_effect_predictor_no_efficiency_lift",
                    "actions_delta": 0.0,
                },
                _summary_clean(),
            ),
            "A3": (
                {
                    **_a3_clean_bank(),
                    "honest_verdict": "complete: ls20_delta_identified_no_bank",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                },
                _summary_clean(),
            ),
        }
    )
    metrics = mod.measure_integrated_metrics(
        audit=audit,
        a1_artifact=_a1_clean_null(),
        a2_artifact=_a2_clean_success(),
        a4_artifact={**_a4_clean_package(), "live_submittable_level_count": 55, "count_delta": 0},
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True},
        orphan_lint={"passed": True},
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        duration_s=1.0,
    )
    assert artifact["honest_verdict"] == (
        "complete: integration_no_clean_metric_bare_config_kept_honest_null"
    )
    assert artifact["submitted_config_raised_metric_clean"] is False
    assert artifact["action_efficiency_integrated"]["actions_delta_vs_bare"] == 0.0
    assert "No clean upstream lever" in artifact["null_delta_methodology_note"]

    broken = dict(artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["verifier_is_oracle"] = True
    broken["parity_test_green"] = False
    broken["orphan_lint_green"] = False
    broken["live_submittable_level_count_integrated"] = 33
    broken["reproducibility_checksum"] = "sha256:bad"
    broken.pop("null_delta_methodology_note")
    errors = mod.artifact_schema_errors(broken)
    assert "honest_verdict_terminal_prefix" in errors
    assert "verifier_is_oracle_false" in errors
    assert "parity_test_green" in errors
    assert "orphan_lint_green" in errors
    assert "live_submittable_level_count_integrated" in errors
    assert "null_delta_methodology_note" in errors
    assert "reproducibility_checksum" in errors

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-4633\n", encoding="utf-8")
    blocked = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        parity_check=lambda _root: {"passed": True},
        orphan_lint=lambda _root: {"passed": True},
        now=lambda: 20.0,
    )
    assert blocked["honest_verdict"] == "blocked_a1_artifact_present"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    assert mod._as_float(True, 7.5) == 7.5
    assert mod._as_int(False, 8) == 8
    assert mod._live_status({"stdout": "LIVE re-check: warn"}) == "warn"
    assert mod._success_verdict({"honest_verdict": "success_metric"}) is True
