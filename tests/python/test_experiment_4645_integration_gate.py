"""Tests for Exp 4645 ARC sprint integration gate.

Spec refs: REQ-ARC-WMTE-4645, SCENARIO-ARC-WMTE-4645.
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
MODULE_PATH = REPO / "python" / "carnot" / "experiment_4645_integration_gate.py"
SUBMITTED_AGENT_CONFIG = {
    "policy": "E3AgentPolicy",
    "value_weight": 0.0,
    "target_levels": 3,
    "frame_change_predictor_enabled": True,
    "frame_change_ranking_mode": "persistent_aem_plus_optional_cnn",
    "action_effect_expansion_prior_enabled": True,
    "action_effect_expansion_prior_mode": "persistent_aem_plus_optional_cnn_frontier_prior",
    "goal_energy_enabled": True,
    "goal_energy_source": "exp4020_graded_goal_satisfaction_energy",
    "goal_energy_alpha": 0.9,
    "goal_energy_beta": 0.1,
    "live_submit_package_path": "results/experiment_4643_submission_package_operator_resubmit.json",
    "live_submit_source": "experiment_4643_refresh_submission_package",
}


def _load_mod() -> Any:
    spec = importlib.util.spec_from_file_location(
        "experiment_4645_integration_gate_under_test", MODULE_PATH
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


def _summary_clean() -> dict[str, Any]:
    return {"returncode": 0, "live_status": "clean", "stdout": "LIVE re-check: clean"}


def _summary_critical() -> dict[str, Any]:
    return {"returncode": 2, "live_status": "CRITICAL", "stdout": "LIVE re-check: CRITICAL"}


def _a1_null() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: goal_energy_no_live_lift_honest_null_gap_sharpened",
        "flagged_adversarial": None,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "uniform_energy_ablation_passed": True,
        "live_solve_rate_goal_energy": 0.04,
        "live_solve_rate_baseline": 0.04,
        "solve_rate_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "median_actions_to_win_delta": 0.0,
        "chosen_submitted_config": "unchanged",
    }


def _a1_success_uniform_failed() -> dict[str, Any]:
    payload = dict(_a1_null())
    payload.update(
        {
            "honest_verdict": "success: goal_energy_live_generation_solverate_up_1",
            "uniform_energy_ablation_passed": False,
            "live_solve_rate_goal_energy": 0.08,
            "solve_rate_delta": 0.04,
            "first_win_rate_delta": 0.04,
        }
    )
    return payload


def _a2_null() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: action_effect_expansion_prior_no_deeper_solve_honest_null_gap_sharpened"
        ),
        "flagged_adversarial": None,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "live_solve_rate_expansion": 0.0,
        "live_solve_rate_ranker_baseline": 0.0,
        "solve_rate_delta": 0.0,
        "depth_of_live_solve_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "median_actions_to_win_expansion": None,
        "median_actions_to_win_ranker_baseline": None,
    }


def _a2_success() -> dict[str, Any]:
    payload = dict(_a2_null())
    payload.update(
        {
            "honest_verdict": "success: action_effect_expansion_prior_live_deeper_solve_1",
            "live_solve_rate_expansion": 0.2,
            "live_solve_rate_ranker_baseline": 0.1,
            "solve_rate_delta": 0.1,
            "depth_of_live_solve_delta": 1.0,
            "first_win_rate_delta": 0.0,
            "median_actions_to_win_expansion": 4.0,
            "median_actions_to_win_ranker_baseline": 7.0,
        }
    )
    return payload


def _a3_bank() -> dict[str, Any]:
    return {
        "honest_verdict": "success: ft09_L3_offline_reproduced",
        "flagged_adversarial": None,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "reproduction_gate": {"game": "ft09", "claimed_level": 3, "reproduced": True},
    }


def _a4_package() -> dict[str, Any]:
    return {
        "honest_verdict": "success: package_refreshed_live_submittable_57_above_33",
        "flagged_adversarial": None,
        "live_submittable_level_count": 57,
        "live_submittable_count_prev": 56,
        "count_delta": 1,
        "ready_for_operator_submit": True,
        "refreshed_package_path": "results/experiment_4643_submission_package_operator_resubmit.json",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4645_spec_declares_integration_gate_contract() -> None:
    """REQ-ARC-WMTE-4645: OpenSpec declares the 4645 artifact schema."""

    mod = _load_mod()
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4645" in spec
    assert "SCENARIO-ARC-WMTE-4645" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4645_audits_uniform_gate_and_quarantines_nulls() -> None:
    """REQ-ARC-WMTE-4645: A1 needs success, controls, and uniform ablation."""

    mod = _load_mod()
    audit = mod.audit_upstream_levers(
        {
            "A1": (_a1_null(), _summary_clean()),
            "A2": (_a2_success(), _summary_critical()),
            "A3": (_a3_bank(), _summary_clean()),
        }
    )

    assert audit["levers_integrated"] == ["A3_level_bank_refreshed_package"]
    assert audit["upstream_lever_audit"]["A1"]["reason"] == "honest_verdict_not_success"
    assert audit["upstream_lever_audit"]["A2"]["reason"] == "flagged_adversarial"
    assert audit["upstream_lever_audit"]["A3"]["reason"] == "admitted_clean_metric_raiser"
    assert audit["flagged_artifacts_excluded"] == [
        {
            "lever": "A2",
            "reason": "flagged_adversarial",
            "live_status": "CRITICAL",
            "honest_verdict": _a2_success()["honest_verdict"],
        }
    ]

    uniform_failed = mod.audit_upstream_levers(
        {
            "A1": (_a1_success_uniform_failed(), _summary_clean()),
            "A2": (_a2_null(), _summary_clean()),
            "A3": ({**_a3_bank(), "offline_reproduced": False}, _summary_clean()),
        }
    )
    assert uniform_failed["upstream_lever_audit"]["A1"]["reason"] == (
        "uniform_energy_ablation_failed"
    )
    assert uniform_failed["flagged_artifacts_excluded"] == [
        {
            "lever": "A1",
            "reason": "uniform_energy_ablation_failed",
            "live_status": "clean",
            "honest_verdict": _a1_success_uniform_failed()["honest_verdict"],
        },
        {
            "lever": "A3",
            "reason": "positive_control_failed",
            "live_status": "clean",
            "honest_verdict": _a3_bank()["honest_verdict"],
        },
    ]


def test_scenario_arc_wmte_4645_builds_package_success_with_null_deltas() -> None:
    """SCENARIO-ARC-WMTE-4645: A3 package win ships while A1/A2 stay honest nulls."""

    mod = _load_mod()
    audit = mod.audit_upstream_levers(
        {
            "A1": (_a1_null(), _summary_clean()),
            "A2": (_a2_null(), _summary_clean()),
            "A3": (_a3_bank(), _summary_clean()),
        }
    )
    metrics = mod.measure_integrated_metrics(
        audit=audit,
        a1_artifact=_a1_null(),
        a2_artifact=_a2_null(),
        a4_artifact=_a4_package(),
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
        artifact["honest_verdict"] == "success: integrated_live_submittable_raised_config_shipped"
    )
    assert artifact["levers_integrated"] == ["A3_level_bank_refreshed_package"]
    assert artifact["live_solve_rate_integrated"] == 0.04
    assert artifact["live_solve_rate_delta_vs_bare"] == 0.0
    assert artifact["live_multi_level_solve_rate_integrated"] == 0.0
    assert artifact["live_multi_level_solve_rate_delta_vs_bare"] == 0.0
    assert artifact["live_submittable_level_count_integrated"] == 57
    assert artifact["live_submittable_delta_vs_baseline"] == 1
    assert artifact["submitted_agent_config"]["live_submit_package_path"] == (
        "results/experiment_4643_submission_package_operator_resubmit.json"
    )
    assert artifact["submitted_config_expected_patch"]["live_submit_source"] == (
        "experiment_4643_refresh_submission_package"
    )
    assert "live_solve_rate_delta_vs_bare" in artifact["null_delta_methodology_note"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4645_run_writes_artifact_with_summarize_and_gates(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SCENARIO-ARC-WMTE-4645: run reads A1/A2/A3 via summarize and writes JSON."""

    mod = _load_mod()
    _install_submitted_config_stub(monkeypatch)

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-4645\n", encoding="utf-8")
    payloads = {
        mod.A1_RELATIVE_PATH: _a1_null(),
        mod.A2_RELATIVE_PATH: _a2_null(),
        mod.A3_RELATIVE_PATH: _a3_bank(),
        mod.A4_RELATIVE_PATH: _a4_package(),
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
        "experiment_4640_goal_energy_generation_live.json",
        "experiment_4641_action_effect_expansion_prior_live.json",
        "experiment_4642_levelup_selfplay.json",
    ]
    assert artifact["preconditions_checked"]["a1_artifact_present"] is True
    assert artifact["preconditions_checked"]["a4_artifact_present"] is True
    assert artifact["parity_test_green"] is True
    assert artifact["orphan_lint_green"] is True

    blocked = mod.run(
        root=tmp_path / "missing",
        offline_arcade_checker=lambda: True,
        parity_check=lambda _root: {"passed": True},
        orphan_lint=lambda _root: {"passed": True},
        now=lambda: 20.0,
    )
    assert blocked["honest_verdict"] == "blocked_agents_md_read"
    assert (tmp_path / "missing" / mod.RESULT_RELATIVE_PATH).exists()
