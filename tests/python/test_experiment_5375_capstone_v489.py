"""Tests for Exp 5375 .489 capstone synthesis.

Spec refs: REQ-CAPSTONE-5375, SCENARIO-CAPSTONE-5375,
SCENARIO-CAPSTONE-5375-MISSING-OR-SKIPPED-INPUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

from carnot import experiment_5375_capstone_v489 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _copy_available_inputs(root: Path) -> None:
    for relative in (*mod.EXPECTED_ARTIFACT_PATHS, *mod.EXTRA_AVAILABLE_ARTIFACT_PATHS):
        source = REPO / relative
        if not source.exists():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)


def test_req_capstone_5375_spec_declares_strict_closeout_contract() -> None:
    """REQ-CAPSTONE-5375: OpenSpec anchors the .489 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5375") :]

    for marker in (
        "REQ-CAPSTONE-5375",
        "SCENARIO-CAPSTONE-5375",
        "SCENARIO-CAPSTONE-5375-MISSING-OR-SKIPPED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        "research-roadmap-next.yaml",
        "results/experiment_5367_constraint_tax_tool_action_panel_v2_v489.json",
        "failed structured protocol gate",
        "text-only or tautological token/internal-feature evidence",
        "CPU-only p-bit diagnostics",
        "honest-null ARC attempt",
        "continuity-only hardware receipts",
        "`hardware_speedup_claim=false`",
    ):
        assert marker in section


def test_scenario_capstone_5375_builds_default_artifact_without_overclaiming() -> None:
    """SCENARIO-CAPSTONE-5375: checked-in .489 artifacts produce honest gates."""

    artifact = mod.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit capstone synthesis", "outcome": "passed"}],
    )

    assert artifact["status"] == "complete"
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["artifacts_expected"] == list(mod.EXPECTED_ARTIFACT_PATHS)
    assert "research-roadmap-next.yaml" in artifact["artifacts_missing"]
    assert (
        "results/experiment_5367_constraint_tax_tool_action_panel_v2_v489.json"
        in artifact["artifacts_missing"]
    )
    assert (
        "results/experiment_5367_v489_constraint_tax_tool_action_panel_v2.json"
        in artifact["artifacts_found"]
    )

    assert artifact["grammar_budget_protocol_ready"] is True
    assert artifact["structured_protocol_clean"] is False
    assert artifact["constraint_tax_panel_ready"] is False
    assert artifact["budget_curated_memory_ready"] is True
    assert artifact["continuous_self_learning_budget_scaleup_ready"] is True
    assert artifact["overwrite_solver_guidance_ready"] is True
    assert artifact["boundary_exchange_schedule_ready"] is True
    assert artifact["token_feature_gate_ready"] is True
    assert artifact["future_token_signal_allowed"] is False
    assert artifact["arc_new_level_banked"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["continuous_self_learning_requirement_satisfied"] is True

    phases = {row["lane"]: row for row in artifact["phase_outcomes"]}
    assert phases["grammar_structured_sota"]["outcome"] == "blocked_structured_protocol_clean_false"
    assert phases["constraint_tax"]["outcome"] == "blocked_or_skipped"
    assert phases["budget_curated_self_learning"]["outcome"] == "ready"
    assert phases["solver_guidance"]["outcome"] == "ready_solver_authoritative"
    assert phases["pbit_boundary_exchange"]["claim_boundary"] == "cpu_simulation_only_no_speedup"
    assert phases["token_internal_feature_gate"]["outcome"] == "retire_until_backend_features"
    assert phases["arc"]["outcome"] == "honest_null_no_new_level_banked"
    assert phases["hardware"]["claim_boundary"] == "continuity_receipts_only_no_speedup"

    assert artifact["ready_gates_for_next_milestone"] == [
        "grammar_budget_protocol_preflight",
        "budget_curated_memory_governance",
        "continuous_self_learning_budget_scaleup",
        "overwrite_solver_guidance",
        "boundary_exchange_schedule_cpu_diagnostic",
        "token_feature_precondition_gate_as_retirement_guard",
    ]
    first_action = artifact["next_milestone_recommendations"][0]
    assert first_action["action"] == "repair_live_structured_protocol_clean_gate"
    assert "no_cpu_only_sota_headline" in first_action["guardrails"]

    blocked = {row["lane"]: row for row in artifact["retired_or_blocked_lanes"]}
    assert blocked["constraint_tax_panel"]["state"] == "blocked"
    assert blocked["external_text_scorer_reopening"]["state"] == "retired_no_go"
    assert blocked["cpu_only_sota_headline"]["state"] == "retired_no_go"
    assert blocked["token_internal_feature_signal"]["state"] == "retired_until_backend_features"
    assert blocked["duplicate_or_offline_arc_solve"]["state"] == "retired_no_go"
    assert blocked["hardware_speedup_claim"]["state"] == "blocked_on_authenticated_evidence"

    assert artifact["no_go_rules_preserved"] == {
        "external_text_scorer_reopened": False,
        "cpu_only_sota_headline": False,
        "duplicate_offline_arc_solve": False,
        "hardware_speedup_without_authenticated_evidence": False,
    }
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_capstone_5375_run_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5375: run writes deterministic capstone JSON."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit run", "outcome": "passed"}]

    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact == mod.build_artifact(root=REPO, tests_run=tests_run)


def test_scenario_capstone_5375_missing_or_skipped_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5375-MISSING-OR-SKIPPED-INPUT: gaps never imply success."""

    _copy_available_inputs(tmp_path)
    (
        tmp_path / "results/experiment_5369_budgeted_continuous_self_learning_scaleup_v489.json"
    ).unlink()
    (tmp_path / "results/experiment_5368_budget_curated_memory_governance_v489.json").write_text(
        "[]",
        encoding="utf-8",
    )
    (tmp_path / "results/experiment_5370_overwrite_solver_guidance_matrix_v489.json").write_text(
        "{",
        encoding="utf-8",
    )
    expected_5367 = (
        tmp_path / "results/experiment_5367_constraint_tax_tool_action_panel_v2_v489.json"
    )
    expected_5367.write_text(
        json.dumps(
            {
                "status": "blocked",
                "constraint_tax_panel_ready": True,
                "blocked_at_layer": "conductor_pre_gate",
                "honest_verdict": "blocked_gate_check_failed",
            }
        ),
        encoding="utf-8",
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["status"] == "complete"
    assert artifact["budget_curated_memory_ready"] is False
    assert artifact["continuous_self_learning_budget_scaleup_ready"] is False
    assert artifact["continuous_self_learning_requirement_satisfied"] is False
    assert artifact["overwrite_solver_guidance_ready"] is False
    assert artifact["constraint_tax_panel_ready"] is False
    assert (
        "results/experiment_5369_budgeted_continuous_self_learning_scaleup_v489.json"
        in artifact["artifacts_missing"]
    )
    assert (
        "results/experiment_5370_overwrite_solver_guidance_matrix_v489.json"
        in artifact["artifacts_unreadable"]
    )
    unreadable = {row["path"]: row for row in artifact["artifact_read_errors"]}
    assert unreadable["results/experiment_5370_overwrite_solver_guidance_matrix_v489.json"][
        "classification"
    ].startswith("malformed_json")
    assert (
        unreadable["results/experiment_5368_budget_curated_memory_governance_v489.json"][
            "classification"
        ]
        == "not_json_object"
    )


def test_capstone_5375_value_unwraps_principled_and_bare_fields() -> None:
    """SCENARIO-CAPSTONE-5375: principle wrappers do not hide source booleans."""

    assert mod.value_of({"principle": "why", "value": False}) is False
    assert mod.value_of(True) is True
    assert mod._source_status(None) == "missing"
    assert mod._source_verdict(None) == "missing"
