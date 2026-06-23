"""Tests for Exp 4621 ARC sprint integration gate.

Spec refs: REQ-ARC-WMTE-4621, SCENARIO-ARC-WMTE-4621.
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
MODULE_PATH = REPO / "python" / "carnot" / "experiment_4621_integration_gate.py"
SUBMITTED_AGENT_CONFIG = {"policy": "E3AgentPolicy", "value_weight": 0.0, "target_levels": 3}


def _load_mod():
    spec = importlib.util.spec_from_file_location("experiment_4621_integration_gate_under_test", MODULE_PATH)
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


def _a2_flagged_null() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened",
        "flagged_adversarial": True,
        "bare_and_linear_controls_passed": True,
        "false_negative_risk_checked": True,
        "first_win_rate_graduated": 0.04,
        "first_win_rate_linear_baseline": 0.04,
        "first_win_rate_bare": 0.04,
        "first_win_delta": 0.0,
        "median_actions_to_first_levelup_graduated": 20.0,
        "median_actions_to_first_levelup_linear_baseline": 20.0,
        "median_actions_to_first_levelup_bare": 20.0,
        "actions_delta": 0.0,
        "solve_rate_graduated": 0.04,
        "solve_rate_linear_baseline": 0.04,
        "solve_rate_bare": 0.04,
        "value_weight_used": 1e-6,
        "chosen_submitted_config": "unchanged",
    }


def _a2_clean_success() -> dict[str, Any]:
    return {
        "honest_verdict": "success: spatial_value_head_graduated_live_first_win_up_1",
        "flagged_adversarial": None,
        "bare_and_linear_controls_passed": True,
        "false_negative_risk_checked": True,
        "first_win_rate_graduated": 0.12,
        "first_win_rate_linear_baseline": 0.04,
        "first_win_rate_bare": 0.04,
        "first_win_delta": 0.08,
        "median_actions_to_first_levelup_graduated": 12.0,
        "median_actions_to_first_levelup_linear_baseline": 20.0,
        "median_actions_to_first_levelup_bare": 20.0,
        "actions_delta": 8.0,
        "offline_to_live_transfer_ratio": 3.0,
        "offline_to_live_transfer_ratio_baseline": 1.0,
        "value_weight_used": 1e-6,
        "chosen_submitted_config": {
            "value_head": "SpatialValueNet",
            "value_mode": "decision_point_cached_tiebreak",
            "bounded_value_weight": 1e-6,
        },
    }


def _a3_no_bank() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: sk48_delta_identified_no_bank",
        "flagged_adversarial": None,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "reproduction_gate": {
            "game": "sk48",
            "claimed_level": 1,
            "reached_level": 1,
            "reproduced": True,
        },
    }


def _a3_clean_bank() -> dict[str, Any]:
    return {
        "honest_verdict": "success: sk48_L2_offline_reproduced",
        "flagged_adversarial": None,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "registry_updated": True,
        "reproduction_gate": {
            "game": "sk48",
            "claimed_level": 2,
            "reached_level": 2,
            "reproduced": True,
        },
    }


def _a4_clean_package() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: package_refreshed_unchanged_depth.",
        "flagged_adversarial": None,
        "live_submittable_level_count": 55,
        "live_submittable_count_prev": 55,
        "count_delta": 0,
        "ready_for_operator_submit": True,
        "refreshed_package_path": "results/experiment_4619_submission_package_operator_resubmit.json",
        "offline_reproduced": True,
    }


def _summary_clean() -> dict[str, Any]:
    return {"returncode": 0, "live_status": "clean", "stdout": "LIVE re-check: clean"}


def _summary_critical() -> dict[str, Any]:
    return {"returncode": 2, "live_status": "CRITICAL", "stdout": "LIVE re-check: CRITICAL"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4621_spec_declares_integration_gate_contract() -> None:
    """REQ-ARC-WMTE-4621: OpenSpec declares the 4621 artifact schema."""

    mod = _load_mod()

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4621" in spec
    assert "SCENARIO-ARC-WMTE-4621" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4621_quarantines_flagged_and_non_success_upstreams() -> None:
    """REQ-ARC-WMTE-4621: flagged A2 and non-success A3 metrics are not aggregated."""

    mod = _load_mod()

    audit = mod.audit_upstream_levers(
        {
            "A2": (_a2_flagged_null(), _summary_critical()),
            "A3": (_a3_no_bank(), _summary_clean()),
            "A4": (_a4_clean_package(), _summary_clean()),
        }
    )

    assert audit["levers_integrated"] == []
    assert audit["submitted_config_raised_metric_clean"] is False
    assert audit["flagged_artifacts_excluded"] == [
        {
            "lever": "A2",
            "reason": "flagged_adversarial",
            "live_status": "CRITICAL",
            "honest_verdict": _a2_flagged_null()["honest_verdict"],
        }
    ]
    assert audit["upstream_lever_audit"]["A2"]["integrated"] is False
    assert audit["upstream_lever_audit"]["A2"]["reason"] == "flagged_adversarial"
    assert audit["upstream_lever_audit"]["A3"]["reason"] == "honest_verdict_not_success"
    assert audit["upstream_lever_audit"]["A4"]["reason"] == "package_metric_only"


def test_scenario_arc_wmte_4621_builds_honest_null_artifact() -> None:
    """SCENARIO-ARC-WMTE-4621: artifact reports required null metrics and checksum."""

    mod = _load_mod()

    audit = mod.audit_upstream_levers(
        {
            "A2": (_a2_flagged_null(), _summary_critical()),
            "A3": (_a3_no_bank(), _summary_clean()),
            "A4": (_a4_clean_package(), _summary_clean()),
        }
    )
    metrics = mod.measure_integrated_metrics(
        audit=audit,
        a2_artifact=_a2_flagged_null(),
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

    assert artifact["honest_verdict"] == (
        "complete: integration_no_clean_metric_bare_config_kept_honest_null"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["levers_integrated"] == []
    assert artifact["offline_to_live_transfer_ratio_integrated"] == 0.0
    assert artifact["first_win_rate_integrated"] == 0.0
    assert artifact["median_actions_to_first_levelup_integrated"] is None
    assert artifact["live_submittable_level_count_integrated"] == 55
    assert artifact["submitted_config_raised_metric_clean"] is False
    assert artifact["parity_test_green"] is True
    assert artifact["orphan_lint_green"] is True
    assert artifact["config_action"] == "unchanged_bare_config_kept"
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4621_run_writes_artifact_with_summarize_and_lint_gates(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SCENARIO-ARC-WMTE-4621: run reads upstreams via summarize and writes JSON."""

    mod = _load_mod()
    _install_submitted_config_stub(monkeypatch)

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-4621\n", encoding="utf-8")
    payloads = {
        mod.A2_RELATIVE_PATH: _a2_flagged_null(),
        mod.A3_RELATIVE_PATH: _a3_no_bank(),
        mod.A4_RELATIVE_PATH: _a4_clean_package(),
    }
    for relative, payload in payloads.items():
        _write_json(tmp_path / relative, payload)

    def summarize(path: Path) -> dict[str, Any]:
        if path.name == "experiment_4617_graduate_spatial_value_head_live.json":
            return _summary_critical()
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
    assert artifact["preconditions_checked"]["a2_artifact_present"] is True
    assert artifact["preconditions_checked"]["a3_artifact_present"] is True
    assert artifact["preconditions_checked"]["a4_artifact_present"] is True
    assert artifact["upstream_lever_audit"]["A4"]["reason"] == "package_metric_only"
    assert artifact["live_submittable_delta_vs_baseline"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4621_success_paths_and_schema_errors() -> None:
    """REQ-ARC-WMTE-4621: clean controls can admit A2/A3 metric raisers."""

    mod = _load_mod()

    audit = mod.audit_upstream_levers(
        {
            "A2": (_a2_clean_success(), _summary_clean()),
            "A3": (_a3_clean_bank(), _summary_clean()),
            "A4": ({**_a4_clean_package(), "live_submittable_level_count": 56}, _summary_clean()),
        }
    )
    metrics = mod.measure_integrated_metrics(
        audit=audit,
        a2_artifact=_a2_clean_success(),
        a4_artifact={**_a4_clean_package(), "live_submittable_level_count": 56},
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

    assert audit["levers_integrated"] == [
        "A2_spatial_value_head_live_path",
        "A3_level_bank_in_refreshed_package",
    ]
    assert artifact["honest_verdict"] == "success: integrated_first_win_raised_config_shipped"
    assert artifact["first_win_rate_integrated"] == 0.12
    assert artifact["first_win_rate_delta_vs_bare"] == 0.08
    assert artifact["offline_to_live_transfer_ratio_integrated"] == 3.0
    assert artifact["offline_to_live_transfer_ratio_delta_vs_baseline"] == 2.0
    assert artifact["actions_delta_vs_bare"] == 8.0
    assert artifact["config_action"] == "ship_clean_metric_levers"
    assert mod.artifact_schema_errors(artifact) == []

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


def test_req_arc_wmte_4621_helper_and_blocked_branches(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-ARC-WMTE-4621: helper edge cases fail closed without fabricating metrics."""

    mod = _load_mod()
    _install_submitted_config_stub(monkeypatch)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    assert mod._as_float(True, 7.5) == 7.5
    assert mod._as_int(False, 8) == 8
    assert mod._as_int("not-an-int", 9) == 9
    assert mod._live_status({"stdout": "LIVE re-check: CRITICAL"}) == "CRITICAL"
    assert mod._live_status({"stdout": "LIVE re-check: warn"}) == "warn"
    assert mod._live_status({"returncode": 1}) == "CRITICAL"
    assert mod._positive_control_passed("A2", {"positive_control_passed": True}) is True
    assert mod._positive_control_passed("unknown", {}) is False
    assert mod._bridge_ratio(
        {"first_win_rate_graduated": 0.1, "first_win_rate_linear_baseline": 0.05}
    ) == (2.0, 1.0)
    assert mod._bridge_ratio({}) == (0.0, 0.0)

    pc_failed = mod.audit_upstream_levers(
        {
            "A2": (
                {
                    "honest_verdict": "success: spatial_value_head_graduated_live_first_win_up_1",
                    "first_win_delta": 1.0,
                },
                _summary_clean(),
            )
        }
    )
    assert pc_failed["upstream_lever_audit"]["A2"]["reason"] == "positive_control_failed"

    no_delta = mod.audit_upstream_levers(
        {
            "A2": (
                {
                    "honest_verdict": "success: spatial_value_head_graduated_live_first_win_up_1",
                    "bare_and_linear_controls_passed": True,
                    "first_win_delta": 0.0,
                    "actions_delta": 0.0,
                },
                _summary_clean(),
            )
        }
    )
    assert no_delta["upstream_lever_audit"]["A2"]["reason"] == "no_positive_metric_delta"

    fallback_metrics = mod.measure_integrated_metrics(
        audit={
            "upstream_lever_audit": {
                "A2": {"integrated": True},
            }
        },
        a2_artifact={"first_win_rate_graduated": 0.1, "first_win_rate_linear_baseline": 0.05},
        a4_artifact={},
    )
    assert fallback_metrics["live_submittable_level_count_integrated"] == mod.LIVE_SUBMITTABLE_BASELINE

    clean_audit = {"submitted_config_raised_metric_clean": True}
    base_metrics = {
        "offline_to_live_transfer_ratio_delta_vs_baseline": 0.0,
        "first_win_rate_delta_vs_bare": 0.0,
        "actions_delta_vs_bare": 0.0,
        "live_submittable_delta_vs_baseline": 0,
    }
    assert mod._verdict(
        clean_audit,
        {**base_metrics, "actions_delta_vs_bare": 2.0},
        parity_green=True,
        orphan_green=True,
    ) == "success: integrated_action_efficiency_raised_config_shipped"
    assert mod._verdict(
        clean_audit,
        {**base_metrics, "live_submittable_delta_vs_baseline": 1},
        parity_green=True,
        orphan_green=True,
    ) == "success: integrated_live_submittable_raised_config_shipped"
    assert mod._verdict(
        clean_audit,
        {**base_metrics, "offline_to_live_transfer_ratio_delta_vs_baseline": 1.0},
        parity_green=True,
        orphan_green=True,
    ) == "success: integrated_offline_to_live_transfer_raised_config_shipped"
    assert mod._verdict(
        clean_audit,
        base_metrics,
        parity_green=True,
        orphan_green=True,
    ).startswith("complete:")

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-4621\n", encoding="utf-8")

    def raises_offline() -> bool:
        raise RuntimeError("offline unavailable")

    checks = mod.check_preconditions(tmp_path, offline_arcade_checker=raises_offline)
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]
    assert checks["blocked_resource"] == "offline_arcade"

    blocked = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        parity_check=lambda _root: {"passed": True},
        orphan_lint=lambda _root: {"passed": True},
        now=lambda: 20.0,
    )
    assert blocked["honest_verdict"] == "blocked_a2_artifact_present"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
