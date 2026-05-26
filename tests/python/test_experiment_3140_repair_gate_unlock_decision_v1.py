"""Tests for Exp 3140 repair-gate unlock decision v1.

Spec refs: REQ-VERIFY-3140, SCENARIO-VERIFY-3140.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import repair_gate_unlock_decision_v1 as mod


REQUIRED_FIELDS = {
    "repair_gate_decision_v1_ready",
    "repair_gate_state",
    "false_accept_rate",
    "false_accept_gate_passed",
    "regression_rows_included",
    "exact_authority_ready",
    "monitor_ledger_ready",
    "selected_repair_rows",
    "repair_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _repair_targets() -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": "repair-arith",
            "fragment_id": "repair-arith:assert_claim",
            "failing_constraint": "claimed_value == computed_value",
            "expected_direction": "replace claimed value 16 with 14",
            "solver_evidence": {
                "authority": "python_ast_literal_evaluator",
                "claimed_value": 16,
                "computed_value": 14,
            },
        },
        {
            "fixture_id": "repair-json",
            "fragment_id": "repair-json:json_document",
            "failing_constraint": "valid_json_document",
            "expected_direction": "produce parseable JSON",
            "solver_evidence": {
                "authority": "python_json_parser",
                "parse_error": "Expecting ',' delimiter",
            },
        },
    ]


def _write_sources(
    root: Path,
    *,
    false_accept_rate: float = 0.0,
    regression_rows_included: bool = True,
    live_model_ready: bool = True,
    exact_ready: bool = True,
    known_false_accepts_blocked: bool = True,
    monitor_ready: bool = True,
    headline_allowed: bool = True,
    repair_manifest_present: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("repair gates need exact authority\n", encoding="utf-8")
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3140\nSCENARIO-VERIFY-3140\n"
        "results/experiment_3140_repair_gate_unlock_decision_v1.json\n",
        encoding="utf-8",
    )

    live_call_count = 6 if live_model_ready else 0
    _write_json(
        root,
        mod.EXP3139_REL_PATH,
        {
            "artifact": "experiment_3139_live_sota_verifier_rerun_v7",
            "live_verifier_rerun_v7_ready": live_model_ready and false_accept_rate <= 0.10,
            "false_accept_rate": false_accept_rate,
            "false_accept_gate_passed": false_accept_rate <= 0.10,
            "regression_rows_included": regression_rows_included,
            "exact_ground_truth_count": 4 if exact_ready else 0,
            "headline_claim_allowed": headline_allowed,
            "live_call_count": live_call_count,
            "selected_model_ids": ["unit/model"] if live_model_ready else [],
            "rerun_rows": [
                {
                    "fixture_id": "fa-arith",
                    "exact_label": "INVALID" if exact_ready else "",
                    "is_regression_row": True,
                    "contract_decision": "abstain",
                },
                {
                    "fixture_id": "clean-valid",
                    "exact_label": "VALID" if exact_ready else "",
                    "is_regression_row": False,
                    "contract_decision": "accept",
                },
            ],
            "inference_substrate": {
                "executes_models": live_model_ready,
                "live_model_calls": live_call_count,
                "selected_model_id": "unit/model" if live_model_ready else None,
                "uses_legacy_small_model_for_headline": False,
            },
            "honest_verdict": "complete: unit rerun",
        },
    )
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "artifact": "experiment_3137_exact_safe_accept_abstain_contract_v1",
            "acceptance_contract_v1_ready": exact_ready,
            "known_false_accept_rows_blocked": known_false_accepts_blocked,
            "replay_false_accept_rate": 0.0,
            "regression_row_set": ["fa-arith"],
            "repair_gate_prerequisites": {
                "require_exact_label_authority": True,
                "require_monitor_ledger_replay_for_live_rows": True,
            },
            "inference_substrate": {"no_live_llm_inference": True},
            "honest_verdict": "complete: exact contract",
        },
    )
    _write_json(
        root,
        mod.EXP3126_REL_PATH,
        {
            "artifact": "experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1",
            "fragment_time_monitor_v1_ready": monitor_ready,
            "ledger_replay_summary": {
                "monitor_event_count": 12 if monitor_ready else 0,
                "ledger_consistency_rate": 1.0 if monitor_ready else 0.0,
            },
            "downstream_repair_constraints": {
                "must_replay_before_repair": monitor_ready,
                "repair_requires_monitor_evidence": monitor_ready,
            },
            "inference_substrate": {"fresh_live_inference_calls": 0},
            "honest_verdict": "complete: monitor ledger",
        },
    )
    _write_json(
        root,
        mod.EXP3125_REL_PATH,
        {
            "artifact": "experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1",
            "prefix_closed_bound_pilot_ready": exact_ready,
            "semantic_coverage": {
                "answer_label_semantics": {"covered": exact_ready, "labels": ["VALID", "INVALID"]}
            },
            "inference_substrate": {"live_model_invoked": False},
            "honest_verdict": "complete: prefix bound",
        },
    )
    manifest_path = mod.REPAIR_TARGET_MANIFEST_REL_PATH
    if repair_manifest_present:
        _write_jsonl(root, manifest_path, _repair_targets())
    _write_json(
        root,
        mod.EXP3115_REL_PATH,
        {
            "artifact": "experiment_3115_explicit_repair_gate_micro_panel_v4",
            "repair_micro_panel_v4_artifact_ready": True,
            "repair_target_manifest_path": manifest_path.as_posix(),
            "repair_target_count": 2 if repair_manifest_present else 0,
            "repair_rows": [],
            "repair_run_executed": False,
            "false_repair_accept_rate": 0.0,
            "inference_substrate": {"repair_run_executed": False},
            "honest_verdict": "complete: prior repair panel",
        },
    )


def test_req_verify_3140_spec_anchor_exists() -> None:
    """REQ-VERIFY-3140: OpenSpec declares the repair-gate decision artifact."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3140" in spec
    assert "SCENARIO-VERIFY-3140" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair_gate_state" in spec
    assert "selected_repair_rows" in spec


def test_scenario_verify_3140_unblocks_with_explicit_repair_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3140: safe rerun evidence unlocks a bounded repair denominator."""

    _write_sources(tmp_path)

    output_path = mod.write_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.25,
        tests_run=["focused-3140"],
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_gate_decision_v1_ready"] is True
    assert artifact["repair_gate_state"] == "unblocked"
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["false_accept_gate_passed"] is True
    assert artifact["regression_rows_included"] is True
    assert artifact["exact_authority_ready"] is True
    assert artifact["monitor_ledger_ready"] is True
    assert artifact["repair_blockers"] == []
    assert [row["fixture_id"] for row in artifact["selected_repair_rows"]] == [
        "repair-arith",
        "repair-json",
    ]
    assert artifact["selected_repair_rows"][0]["constraints"]["failing_constraint"] == (
        "claimed_value == computed_value"
    )
    assert artifact["inference_substrate"]["executes_models"] is False
    assert artifact["inference_substrate"]["live_model_calls"] == 0
    assert artifact["inference_substrate"]["repair_calls"] == 0
    assert artifact["tests_run"] == ["focused-3140"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_verify_3140_blocks_false_accepts_before_repair_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3140: false accepts above gate keep repair blocked."""

    _write_sources(tmp_path, false_accept_rate=0.25)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["repair_gate_state"] == "blocked_false_accept"
    assert artifact["false_accept_gate_passed"] is False
    assert artifact["selected_repair_rows"] == []
    assert any("false_accept_rate=0.25" in blocker for blocker in artifact["repair_blockers"])
    assert artifact["honest_verdict"].startswith("blocked_false_accept:")


@pytest.mark.parametrize(
    ("overrides", "expected_state", "blocker_fragment"),
    [
        (
            {"live_model_ready": False},
            "blocked_missing_live_model",
            "bounded live model rerun is missing",
        ),
        (
            {"exact_ready": False},
            "blocked_missing_exact_labels",
            "exact authority is not ready",
        ),
        (
            {"monitor_ready": False},
            "blocked_other",
            "monitor ledger replay is not ready",
        ),
        (
            {"headline_allowed": False},
            "blocked_other",
            "headline_claim_allowed is not true",
        ),
        (
            {"known_false_accepts_blocked": False},
            "blocked_other",
            "known false accepts are not blocked",
        ),
        (
            {"regression_rows_included": False},
            "blocked_other",
            "regression_rows_included is not true",
        ),
        (
            {"repair_manifest_present": False},
            "blocked_other",
            "selected repair row constraints are missing",
        ),
    ],
)
def test_scenario_verify_3140_classifies_blocked_preconditions(
    tmp_path: Path,
    overrides: dict[str, Any],
    expected_state: str,
    blocker_fragment: str,
) -> None:
    """SCENARIO-VERIFY-3140: blocked decisions are actionable and do not select repair rows."""

    _write_sources(tmp_path, **overrides)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_state"] == expected_state
    assert artifact["selected_repair_rows"] == []
    assert any(blocker_fragment in blocker for blocker in artifact["repair_blockers"])
    assert artifact["honest_verdict"].startswith(f"{expected_state}:")


def test_req_verify_3140_validation_rejects_nonterminal_artifact() -> None:
    """REQ-VERIFY-3140: conductor-facing state and verdict fields are mandatory."""

    with pytest.raises(ValueError, match="allowed state"):
        mod.validate_artifact(
            {
                field: True
                for field in REQUIRED_FIELDS
                if field
                not in {
                    "repair_gate_state",
                    "false_accept_rate",
                    "selected_repair_rows",
                    "repair_blockers",
                    "source_artifacts",
                    "inference_substrate",
                    "honest_verdict",
                }
            }
            | {
                "repair_gate_state": "maybe",
                "false_accept_rate": 0.0,
                "selected_repair_rows": [],
                "repair_blockers": [],
                "source_artifacts": [],
                "inference_substrate": {},
                "honest_verdict": "complete: invalid state",
            }
        )


def test_req_verify_3140_guard_helpers_cover_unsafe_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3140: helper guards fail closed on unsafe edge evidence."""

    assert (
        mod.known_false_accepts_blocked(
            {"known_false_accept_rows_blocked": True, "regression_row_set": []},
            {"rerun_rows": []},
        )
        is False
    )
    assert (
        mod.known_false_accepts_blocked(
            {"known_false_accept_rows_blocked": True, "regression_row_set": ["fa"]},
            {"rerun_rows": [{"fixture_id": "fa", "contract_decision": "accept"}]},
        )
        is False
    )

    disqualifiers = mod.headline_disqualifiers(
        {
            "headline_claim_allowed": True,
            "inference_substrate": {"uses_legacy_small_model_for_headline": True},
            "flagged_adversarial": True,
        },
        {"corrigendum_pending": True},
        {},
        {},
        {},
    )
    assert "legacy small model evidence is not headline eligible" in disqualifiers
    assert "exp3139 flagged_adversarial=true" in disqualifiers
    assert "exp3137 corrigendum_pending=true" in disqualifiers

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('\n{"fixture_id": "ok"}\n', encoding="utf-8")
    assert mod.read_jsonl_rows(jsonl_path) == [{"fixture_id": "ok"}]
    assert math.isnan(mod.finite_metric("not-a-number"))
