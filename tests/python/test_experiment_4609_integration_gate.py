"""Tests for Exp 4609 submitted-agent integration consolidation.

Spec refs: REQ-ARC-WMTE-4609, SCENARIO-ARC-WMTE-4609.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _a1_flagged_success() -> dict[str, Any]:
    return {
        "honest_verdict": "success: world_model_trust_energy_pass_rate_up_6_first_win_up",
        "flagged_adversarial": True,
        "world_model_trust_pass_rate_new": 1.0,
        "world_model_trust_pass_rate_binary": 0.0,
        "trust_pass_rate_delta": 1.0,
        "first_win_delta": 1.0,
        "chosen_submitted_config": "enable_world_model_trust_energy_gate",
    }


def _a2_flagged_null() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
        "flagged_adversarial": True,
        "first_win_rate_integrated": 0.04,
        "first_win_rate_bare": 0.04,
        "first_win_delta": 0.0,
        "median_actions_to_first_levelup_integrated": 20.0,
        "median_actions_to_first_levelup_bare": 20.0,
        "actions_delta": 0.0,
        "chosen_submitted_config": "unchanged",
    }


def _a3_no_bank() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: dc22_delta_identified_no_bank",
        "flagged_adversarial": None,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "reproduction_gate": {"reproduced": False, "reached_level": 1, "claimed_level": 2},
    }


def _a4_clean_package() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: package_refreshed_unchanged_depth.",
        "flagged_adversarial": None,
        "live_submittable_level_count": 55,
        "live_submittable_count_prev": 55,
        "count_delta": 0,
        "ready_for_operator_submit": True,
        "refreshed_package_path": "results/experiment_4607_submission_package_operator_resubmit.json",
        "offline_reproduced": True,
    }


def _summary_clean() -> dict[str, Any]:
    return {"returncode": 0, "live_status": "clean", "stdout": "clean"}


def _summary_critical() -> dict[str, Any]:
    return {"returncode": 2, "live_status": "CRITICAL", "stdout": "LIVE re-check: CRITICAL"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4609_spec_declares_integration_gate_contract() -> None:
    """REQ-ARC-WMTE-4609: OpenSpec declares the consolidation artifact schema."""

    from carnot import experiment_4609_integration_gate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4609" in spec
    assert "SCENARIO-ARC-WMTE-4609" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4609_quarantines_flagged_and_non_success_upstreams() -> None:
    """REQ-ARC-WMTE-4609: flagged or non-success upstream metrics are not aggregated."""

    from carnot import experiment_4609_integration_gate as mod

    audit = mod.audit_upstream_levers(
        {
            "A1": (_a1_flagged_success(), _summary_critical()),
            "A2": (_a2_flagged_null(), _summary_critical()),
            "A3": (_a3_no_bank(), _summary_clean()),
            "A4": (_a4_clean_package(), _summary_clean()),
        }
    )

    assert audit["levers_integrated"] == []
    assert audit["submitted_config_raised_metric_clean"] is False
    assert {row["lever"] for row in audit["flagged_artifacts_excluded"]} == {"A1", "A2"}
    assert audit["upstream_lever_audit"]["A1"]["integrated"] is False
    assert audit["upstream_lever_audit"]["A1"]["reason"] == "flagged_adversarial"
    assert audit["upstream_lever_audit"]["A3"]["reason"] == "honest_verdict_not_success"


def test_scenario_arc_wmte_4609_builds_honest_null_artifact() -> None:
    """SCENARIO-ARC-WMTE-4609: artifact reports required null metrics and checksum."""

    from carnot import experiment_4609_integration_gate as mod

    audit = mod.audit_upstream_levers(
        {
            "A1": (_a1_flagged_success(), _summary_critical()),
            "A2": (_a2_flagged_null(), _summary_critical()),
            "A3": (_a3_no_bank(), _summary_clean()),
            "A4": (_a4_clean_package(), _summary_clean()),
        }
    )
    metrics = mod.measure_integrated_metrics(
        audit=audit,
        a4_artifact=_a4_clean_package(),
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True, "command": "pytest parity"},
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == (
        "complete: integration_no_clean_metric_bare_config_kept_honest_null"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["levers_integrated"] == []
    assert artifact["world_model_trust_pass_rate_integrated"] == 0.0
    assert artifact["first_win_rate_integrated"] == 0.0
    assert artifact["median_actions_to_first_levelup_integrated"] is None
    assert artifact["live_submittable_level_count_integrated"] == 55
    assert artifact["submitted_config_raised_metric_clean"] is False
    assert artifact["parity_test_green"] is True
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4609_run_writes_artifact_with_summarize_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4609: run reads upstreams via summarize gate and writes JSON."""

    from carnot import experiment_4609_integration_gate as mod

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-4609\n", encoding="utf-8")
    payloads = {
        mod.A1_RELATIVE_PATH: _a1_flagged_success(),
        mod.A2_RELATIVE_PATH: _a2_flagged_null(),
        mod.A3_RELATIVE_PATH: _a3_no_bank(),
        mod.A4_RELATIVE_PATH: _a4_clean_package(),
    }
    for relative, payload in payloads.items():
        _write_json(tmp_path / relative, payload)

    def summarize(path: Path) -> dict[str, Any]:
        if path.name in {
            "experiment_4604_world_model_trust_energy.json",
            "experiment_4605_live_integration_scored_agent.json",
        }:
            return _summary_critical()
        return _summary_clean()

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        summarize_runner=summarize,
        parity_check=lambda _root: {"passed": True, "command": "pytest parity"},
        now=lambda: 10.0,
    )

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    assert artifact["preconditions_checked"]["a4_artifact_present"] is True
    assert artifact["upstream_lever_audit"]["A4"]["reason"] == "package_metric_only"
    assert artifact["live_submittable_delta_vs_baseline"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4609_success_path_and_schema_errors() -> None:
    """REQ-ARC-WMTE-4609: clean positive controls can admit a metric raiser."""

    from carnot import experiment_4609_integration_gate as mod

    clean_a1 = dict(
        _a1_flagged_success(),
        flagged_adversarial=None,
        positive_control_passed=True,
    )
    audit = mod.audit_upstream_levers(
        {
            "A1": (clean_a1, _summary_clean()),
            "A2": (_a2_flagged_null(), _summary_critical()),
            "A3": (_a3_no_bank(), _summary_clean()),
            "A4": (_a4_clean_package(), _summary_clean()),
        }
    )
    metrics = mod.measure_integrated_metrics(
        audit=audit,
        a4_artifact=_a4_clean_package(),
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        audit=audit,
        metrics=metrics,
        parity_test={"passed": True},
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        duration_s=1.0,
    )

    assert audit["levers_integrated"] == ["A1_world_model_trust_energy_gate"]
    assert artifact["honest_verdict"] == "success: integrated_world_model_trust_raised_config_shipped"
    assert artifact["world_model_trust_pass_rate_integrated"] == 1.0
    assert artifact["world_model_trust_pass_rate_delta_vs_baseline"] == 1.0
    assert mod.artifact_schema_errors(artifact) == []

    broken = dict(artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["verifier_is_oracle"] = True
    broken["parity_test_green"] = False
    broken["live_submittable_level_count_integrated"] = 33
    broken["reproducibility_checksum"] = "sha256:bad"
    broken.pop("null_delta_methodology_note")
    errors = mod.artifact_schema_errors(broken)
    assert "honest_verdict_terminal_prefix" in errors
    assert "verifier_is_oracle_false" in errors
    assert "parity_test_green" in errors
    assert "live_submittable_level_count_integrated" in errors
    assert "null_delta_methodology_note" in errors
    assert "reproducibility_checksum" in errors


def test_req_arc_wmte_4609_helper_and_blocked_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4609: helper edge cases fail closed without fabricating metrics."""

    from carnot import experiment_4609_integration_gate as mod

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    assert mod._as_float(True, 7.5) == 7.5
    assert mod._as_int(False, 8) == 8
    assert mod._as_int("not-an-int", 9) == 9
    assert mod._live_status({"stdout": "LIVE re-check: CRITICAL"}) == "CRITICAL"
    assert mod._live_status({"stdout": "LIVE re-check: warn"}) == "warn"
    assert mod._live_status({"returncode": 1}) == "CRITICAL"
    assert mod._positive_control_passed("unknown", {}) is False

    pc_failed = mod.audit_upstream_levers(
        {
            "A1": (
                {
                    "honest_verdict": "success: world_model_trust_energy_pass_rate_up",
                    "trust_pass_rate_delta": 1.0,
                },
                _summary_clean(),
            )
        }
    )
    assert pc_failed["upstream_lever_audit"]["A1"]["reason"] == "positive_control_failed"

    no_delta = mod.audit_upstream_levers(
        {
            "A1": (
                {
                    "honest_verdict": "success: world_model_trust_energy_pass_rate_up",
                    "binary_gate_control_passed": True,
                    "trust_pass_rate_delta": 0.0,
                },
                _summary_clean(),
            )
        }
    )
    assert no_delta["upstream_lever_audit"]["A1"]["reason"] == "no_positive_metric_delta"

    a2_audit = mod.audit_upstream_levers(
        {
            "A2": (
                {
                    "honest_verdict": "success: live_integration_scored_first_win_up_1",
                    "bare_control_passed": True,
                    "first_win_delta": 0.25,
                },
                _summary_clean(),
            ),
            "A4": (_a4_clean_package(), _summary_clean()),
        }
    )
    metrics = mod.measure_integrated_metrics(
        audit=a2_audit,
        a4_artifact={},
        submitted_agent_config={
            "first_win_rate_integrated": 0.5,
            "median_actions_to_first_levelup_integrated": 7,
            "median_actions_to_first_levelup_bare": 10,
        },
    )
    assert metrics["first_win_rate_integrated"] == 0.5
    assert metrics["median_actions_to_first_levelup_integrated"] == 7
    assert metrics["live_submittable_level_count_integrated"] == mod.LIVE_SUBMITTABLE_BASELINE

    live_metrics = dict(metrics, live_submittable_delta_vs_baseline=2)
    assert (
        mod._verdict({"submitted_config_raised_metric_clean": True}, live_metrics, True)
        == "success: integrated_first_win_raised_config_shipped"
    )
    only_live_metrics = dict(
        metrics,
        first_win_rate_delta_vs_bare=0.0,
        live_submittable_delta_vs_baseline=2,
    )
    assert (
        mod._verdict({"submitted_config_raised_metric_clean": True}, only_live_metrics, True)
        == "success: integrated_live_submittable_raised_config_shipped"
    )
    zero_metrics = dict(only_live_metrics, live_submittable_delta_vs_baseline=0)
    assert mod._verdict({"submitted_config_raised_metric_clean": True}, zero_metrics, True).startswith(
        "complete:"
    )

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-4609\n", encoding="utf-8")

    def raises_offline() -> bool:
        raise RuntimeError("offline unavailable")

    checks = mod.check_preconditions(tmp_path, offline_arcade_checker=raises_offline)
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]
    assert checks["blocked_resource"] == "offline_arcade"

    blocked = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        parity_check=lambda _root: {"passed": False},
        now=lambda: 20.0,
    )
    assert blocked["honest_verdict"] == "blocked_a1_artifact_present"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
