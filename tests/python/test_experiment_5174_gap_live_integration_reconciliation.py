"""Tests for Exp5174 GAP-LIVE-INTEGRATION reconciliation.

Spec refs: REQ-CAPSTONE-5174, SCENARIO-CAPSTONE-5174,
SCENARIO-CAPSTONE-5174-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def test_req_capstone_5174_spec_declares_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5174: OpenSpec anchors the reconciliation audit artifact."""

    from carnot import experiment_5174_gap_live_integration_reconciliation_v474 as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-5174" in spec
    assert "SCENARIO-CAPSTONE-5174" in spec
    assert "SCENARIO-CAPSTONE-5174-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_5174_source_claims_are_checked_against_current_lines() -> None:
    """SCENARIO-CAPSTONE-5174: stale claims are resolved from current source lines."""

    from carnot import experiment_5174_gap_live_integration_reconciliation_v474 as mod

    claims = mod.current_source_claims(REPO)

    assert claims["claim_router_dsl_unimported"]["value"] is False
    assert "arc_competition_agent.py:30" in claims["claim_router_dsl_unimported"]["evidence"]
    assert "arc_competition_agent.py:51" in claims["claim_router_dsl_unimported"]["evidence"]
    assert "arc_competition_agent.py:2157" in claims["claim_router_dsl_unimported"]["evidence"]
    assert "arc_competition_agent.py:2164" in claims["claim_router_dsl_unimported"]["evidence"]

    assert claims["claim_target_levels_1"]["value"] is False
    assert "arc_competition_agent.py:88" in claims["claim_target_levels_1"]["evidence"]
    assert "SUBMITTED_TARGET_LEVELS = 3" in claims["claim_target_levels_1"]["evidence"]
    assert "arc_competition_agent.py:3132" in claims["claim_target_levels_1"]["evidence"]

    value_claim = claims["claim_value_weight_0"]
    assert value_claim["value"] is False
    assert value_claim["current_submitted_value_weight"] == 1e-12
    assert "arc_competition_agent.py:83" in value_claim["evidence"]
    assert "arc_competition_agent.py:3131" in value_claim["evidence"]
    assert "tried_nonzero_no_lift" in value_claim["meaningful_distinction"]


def test_scenario_capstone_5174_upstream_artifacts_resolve_value_weight_null() -> None:
    """SCENARIO-CAPSTONE-5174: Exp4652 means tried-and-nulled, not never-wired."""

    from carnot import experiment_5174_gap_live_integration_reconciliation_v474 as mod

    exp4605 = mod.read_exp4605_summary(REPO)
    exp4652 = mod.read_exp4652_summary(REPO)

    assert exp4605["submitted_agent_config"]["target_levels"] == 3
    assert exp4605["submitted_agent_config"]["router_wired"] is True
    assert exp4605["submitted_agent_config"]["world_model_dsl_wired"] is True
    assert exp4605["value_weight_used"] == 1e-12

    assert exp4652["live_baseline_value_weight_zero"]["value_weight"] == 0.0
    assert exp4652["value_weight_set"] == 1e-12
    assert exp4652["first_win_rate_delta"] == 0.0
    assert exp4652["solve_rate_delta"] == 0.0
    assert exp4652["residual_cause_hypothesis"] == "distribution_shift_or_calibration"
    assert "honest no-value null" in exp4652["null_delta_methodology_note"]


def test_scenario_capstone_5174_registry_provenance_audit_counts_banked_paths() -> None:
    """SCENARIO-CAPSTONE-5174: provenance audit answers the mirage-vs-real question."""

    from carnot import experiment_5174_gap_live_integration_reconciliation_v474 as mod

    audit = mod.audit_registry_solve_provenance(REPO)

    assert audit["live_agent_self_discovery_count"] == 4
    assert audit["development_proxy_count"] == 20
    assert audit["out_of_registry_declared_games"] == 24
    assert audit["registry_rows_with_reproducible_levels"] == 25
    assert audit["declared_total_games_mismatch"] is True
    assert audit["excluded_from_declared_24_view"] == ["wa30"]
    assert audit["row_level_counts"]["live_agent_self_discovery"] == 4
    assert audit["row_level_counts"]["development_proxy"] == 21
    assert audit["per_game_basis"]["r11l"]["classified_provenance"] == "live_agent_self_discovery"
    assert audit["per_game_basis"]["bp35"]["classified_provenance"] == "development_proxy"
    assert audit["per_game_basis"]["wa30"]["classified_provenance"] == "development_proxy"


def test_scenario_capstone_5174_lint_and_artifact_are_stable(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5174: artifact records lint output and required fields."""

    from carnot import experiment_5174_gap_live_integration_reconciliation_v474 as mod

    def fake_runner(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout="OK: all solver-like ARC modules are reachable from the live agent path (46 modules in the live closure).\n",
            stderr="",
        )

    lint = mod.run_orphan_lint(REPO, runner=fake_runner)
    artifact = mod.build_artifact(
        REPO,
        orphan_lint_result=lint,
        verifier_gaps_md_updated=True,
        tests_run=["unit fixture"],
        duration_s=1.25,
    )
    errors = mod.artifact_schema_errors(artifact)
    out = mod.write_artifact(artifact, root=tmp_path)
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert lint["value"].startswith("pass")
    assert artifact["honest_verdict"].startswith("complete:")
    assert "original three GAP-LIVE-INTEGRATION claims were stale" in artifact["honest_verdict"]
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["claim_router_dsl_unimported"]["value"] is False
    assert artifact["claim_target_levels_1"]["value"] is False
    assert artifact["claim_value_weight_0"]["value"] is False
    assert artifact["verifier_gaps_md_updated"] is True
    assert artifact["gap_status_recommendation"]["value"] == "re-scoped"
    assert artifact["gap_status_recommendation"]["new_scope"].startswith("banked ARC registry")
    assert loaded == artifact
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert errors == []

