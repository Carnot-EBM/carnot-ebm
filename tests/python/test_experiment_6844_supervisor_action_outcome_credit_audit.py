"""REQ-ARC-6844 supervisor action outcome credit audit tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.agentic import arc_solve_artifact_discipline as discipline
from carnot import experiment_6844_supervisor_action_outcome_credit_audit as exp
from scripts import arc_artifact_lint as arc_lint


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _source_paths(
    tmp_path: Path,
    *,
    inventory: dict[str, Any] | None = None,
    outcomes: dict[str, Any] | None = None,
    supervisor_ab: dict[str, Any] | None = None,
    blocked_receipts: dict[str, Any] | None = None,
    resource_block: dict[str, Any] | None = None,
) -> dict[str, Path]:
    return {
        "experiment_6843": _write_json(
            tmp_path / "results/experiment_6843_live_arc_evidence_stratum_freeze.json",
            inventory or _inventory(),
        ),
        "experiment_6681": _write_json(
            tmp_path / "results/experiment_6681_arc_post_redirect_outcomes.json",
            outcomes or _outcome_artifact(),
        ),
        "experiment_6682": _write_json(
            tmp_path / "results/experiment_6682_arc_held_family_supervisor_ab.json",
            supervisor_ab or _supervisor_ab_artifact(),
        ),
        "experiment_6524": _write_json(
            tmp_path / "results/experiment_6524_arc_supervisor_redirect_generalization.json",
            blocked_receipts or _missing_receipts_artifact(),
        ),
        "experiment_6776": _write_json(
            tmp_path / "results/experiment_6776_arc_shadow_supervisor_accrual.json",
            resource_block or _resource_block_artifact(),
        ),
    }


def _inventory(*, complete: int = 1, eligible_count: int = 1) -> dict[str, Any]:
    rows = [
        {
            "artifact_family": "supervisor_refinement_ledger",
            "budget": 70,
            "game": "tn36",
            "model_id": "unknown",
            "policy": "carnot.agentic.arc_competition_agent.E3AgentPolicy",
            "run_id": "ledger-run",
            "source_path": "ops/arc_supervisor_refinement_ledger.json",
            "stratum_identity": "supervisor|tn36|ledger-run",
            "supervisor_receipt_complete": True,
            "supervisor_state": "applied",
            "tool_loop_state": "off_or_unobserved",
        }
    ]
    return {
        "status": "complete_live_arc_inventory",
        "arc_inventory_complete_score": complete,
        "supervisor_eligible_cells": {
            "count": eligible_count,
            "row_ids": ["supervisor|tn36|ledger-run"] if eligible_count else [],
        },
        "rows": rows if eligible_count else [],
        "unmatched_cell_reasons": [],
        "honest_verdict": "complete_live_arc_inventory",
        "verdict_class": "null",
    }


def _ids(prefix: str) -> dict[str, str]:
    return {
        key: f"sha256:{prefix}{index:062d}"
        for index, key in enumerate(
            ("proposal_id", "application_id", "environment_step_id", "outcome_id"),
            start=1,
        )
    }


def _row(
    *,
    index: int,
    redirect: bool,
    game: str = "tn36",
    run: str = "tn36:0",
    model: str = "model-a",
    policy: str = "policy-a",
    budget: int = 70,
    loop: str = "off_or_unobserved",
    supervisor_mode: str = "applied",
    before: int = 0,
    after: int = 0,
    action_kind: int | str = 6,
    available: list[int | str] | None = None,
    outcome_offset: int = 1,
) -> dict[str, Any]:
    ids = _ids(str(index))
    proposed = {"kind": action_kind, "data": {"x": index, "y": index}}
    applied = {"kind": "RESET", "data": None} if redirect else proposed
    return {
        **ids,
        "lineage": dict(ids),
        "family": game,
        "attempt": 0,
        "episode_id": run,
        "episode_seed": 6681001,
        "model_id": model,
        "policy": policy,
        "budget": budget,
        "tool_loop_state": loop,
        "supervisor_mode": supervisor_mode,
        "action_index": index,
        "decision_sequence": index,
        "outcome_sequence": index + outcome_offset,
        "proposed_action": proposed,
        "applied_action": applied,
        "policy_selected_action": applied,
        "redirect_applied": redirect,
        "redirect_reason": "reset_after_stagnant_repeat" if redirect else None,
        "supervisor_decision": {
            "arm": "reset_after_stagnant_repeat" if redirect else None,
            "fired": redirect,
            "state": "stagnant_repeat" if redirect else "observing",
        },
        "observation_before": {
            "available_actions": available if available is not None else [6],
            "levels_completed": before,
        },
        "observation_after": {
            "available_actions": [6],
            "levels_completed": after,
            "state": "NOT_FINISHED",
        },
        "levels_completed_before": before,
        "levels_completed_after": after,
        "level_change": after > before,
        "reward": {
            "present": False,
            "value": None,
            "synthetic": False,
            "source": "arc_agi.FrameDataRaw.step_return_schema",
        },
        "termination": {
            "terminated": False,
            "truncated": False,
            "state": "NOT_FINISHED",
            "source": "arc_agi.FrameDataRaw.state",
        },
        "return_hash": f"sha256:return{index:058d}",
        "state_hash": f"sha256:state{index:059d}",
        "return_schema": "arc_agi.FrameDataRaw",
        "outcome_status": "returned",
        "live_return": True,
        "fully_joined": True,
        "error": None,
        "action_cost": 1,
    }


def _outcome_artifact(
    *,
    redirect_rows: list[dict[str, Any]] | None = None,
    control_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    redirects = redirect_rows if redirect_rows is not None else [_row(index=10, redirect=True)]
    controls = (
        control_rows
        if control_rows is not None
        else [
            _row(index=1, redirect=False, after=1),
            _row(index=2, redirect=False, action_kind=9, available=[6]),
        ]
    )
    return {
        "status": "complete_arc_outcome_transport_ready",
        "honest_verdict": "complete: exact live outcomes joined",
        "verdict_class": "null",
        "arc_outcome_transport_ready": True,
        "eligible_redirect_outcome_rows": len(redirects),
        "redirect_outcome_rows": redirects,
        "non_redirect_control_rows": controls,
        "canonical_path_receipt": {
            "policy": "policy-a",
            "supervisor": "TraceAutomatonSupervisor.select_action",
            "live_metadata": {
                "episode_rows": [
                    {
                        "family": "tn36",
                        "episode_id": "tn36:0",
                        "actions": 70,
                        "redirects": len(redirects),
                        "controls": len(controls),
                        "lineage_ready": True,
                        "status": "complete",
                    }
                ]
            },
        },
        "gate_check_summary": {"passed": True},
    }


def _supervisor_ab_artifact() -> dict[str, Any]:
    return {
        "status": "complete_arc_supervisor_ab_partial",
        "honest_verdict": "blocked: held-family supervisor A/B is partial",
        "verdict_class": "partial",
        "gate_check_summary": {
            "passed": False,
            "failed_check": "verification_failure",
            "observed": {"failed_test_commands": [".venv/bin/pytest tests/python -q"]},
        },
        "paired_episode_rows": [
            {
                "family": "vc33",
                "matched_unit_id": "vc33:0",
                "transition_utility_delta": -1.0,
                "forbidden_no_headroom": True,
            }
        ],
        "false_intervention_rows": [
            {
                "family": "vc33",
                "matched_unit_id": "vc33:0",
                "action_index": 12,
                "outcome_id": "sha256:partial",
            }
        ],
    }


def _missing_receipts_artifact() -> dict[str, Any]:
    return {
        "status": "blocked_missing_outcome_bearing_live_receipts",
        "honest_verdict": "blocked: missing outcome-bearing live trajectory-supervisor receipts",
        "verdict_class": "blocked",
        "gate_check_summary": {
            "all_gates_passed": True,
            "checks": [
                {
                    "gate": "outcome_bearing_receipts_present_or_blocked",
                    "expected": ">0 or blocked",
                    "observed": 0,
                    "passed": True,
                }
            ],
        },
        "redirect_outcome_rows": [],
        "per_arm_rows": [],
    }


def _resource_block_artifact() -> dict[str, Any]:
    return {
        "status": "complete_blocked_shadow_supervisor_accrual",
        "honest_verdict": "complete_blocked_shadow_supervisor_accrual",
        "verdict_class": "blocked",
        "shadow_supervisor_transport_ready": False,
        "gate_check_summary": {
            "passed": False,
            "failed_check": "exclusive_gpu_without_unrelated_compute",
            "observed": [{"memory_free_mb": 1}],
        },
        "rows": [],
    }


def test_req_6844_spec_precedes_implementation() -> None:
    """REQ-ARC-6844 declares gates, rows, and the required artifact fields."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-6844:") :]
    for marker in (
        "SCENARIO-ARC-6844-ACTION-OUTCOME-JOIN",
        "SCENARIO-ARC-6844-GATES-FAIL-CLOSED",
        "SCENARIO-ARC-6844-STRATA-NOT-POOLED",
        "SCENARIO-ARC-6844-HASHES-NO-SOLVE",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_scenario_6844_action_join_direction_dose_and_invalid_rate(tmp_path: Path) -> None:
    """SCENARIO-ARC-6844-ACTION-OUTCOME-JOIN assigns action dose and outcomes."""

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path),
    )

    assert artifact["status"] == "complete_supervisor_outcome_credit_audit"
    assert artifact["supervisor_effect_eligible_score"] == 1
    assert artifact["solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    rows = artifact["per_game_results"]
    assert [row["row_kind"] for row in rows] == [
        "matched_control",
        "matched_control",
        "eligible_redirect",
    ]
    redirect = next(row for row in rows if row["row_kind"] == "eligible_redirect")
    control = next(row for row in rows if row["transition_progress"] == 1)
    invalid = next(row for row in rows if row["invalid_action"])
    assert redirect["later_exact_outcome"]["direction"] == "abstention"
    assert control["later_exact_outcome"]["direction"] == "progress"
    assert redirect["dose"]["value"] == 1
    assert redirect["dose"]["applied_to_action_identity"] == redirect["action_identity"]
    assert invalid["proposal_validity"]["valid"] is False
    assert artifact["invalid_action_results"][0]["invalid_action_rate"] == pytest.approx(1 / 3)
    assert artifact["headroom_results"][0]["nonzero_headroom"] is True
    assert exp.validate_artifact(artifact) == []


def test_scenario_6844_zero_headroom_blocks_without_dropping_rows(tmp_path: Path) -> None:
    """SCENARIO-ARC-6844-GATES-FAIL-CLOSED blocks zero matched headroom."""

    outcomes = _outcome_artifact(
        redirect_rows=[_row(index=10, redirect=True)],
        control_rows=[_row(index=1, redirect=False)],
    )
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path, outcomes=outcomes),
    )

    assert artifact["status"] == "complete_blocked_supervisor_outcome_credit_audit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["supervisor_effect_eligible_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "headroom_nonzero"
    assert artifact["per_game_results"]
    assert all(row["headroom"]["nonzero_headroom"] is False for row in artifact["per_game_results"])
    assert exp.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutator", "failed_check"),
    [
        (
            lambda row: row.update({"outcome_id": None}),
            "exact_later_outcomes",
        ),
        (
            lambda row: row.update({"outcome_sequence": row["decision_sequence"]}),
            "temporal_order",
        ),
    ],
)
def test_scenario_6844_missing_receipts_and_temporal_order_fail_closed(
    tmp_path: Path,
    mutator: Any,
    failed_check: str,
) -> None:
    """SCENARIO-ARC-6844-GATES-FAIL-CLOSED rejects absent or pre-action outcomes."""

    redirect = _row(index=10, redirect=True)
    mutator(redirect)
    outcomes = _outcome_artifact(redirect_rows=[redirect])

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path, outcomes=outcomes),
    )

    assert artifact["status"] == "complete_blocked_supervisor_outcome_credit_audit"
    assert artifact["gate_check_summary"]["failed_check"] == failed_check
    assert artifact["supervisor_effect_eligible_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_scenario_6844_matching_never_pools_game_or_loop_state(tmp_path: Path) -> None:
    """SCENARIO-ARC-6844-STRATA-NOT-POOLED leaves unlike controls unmatched."""

    outcomes = _outcome_artifact(
        redirect_rows=[_row(index=10, redirect=True, game="tn36", loop="off_or_unobserved")],
        control_rows=[
            _row(index=1, redirect=False, game="tr87", loop="off_or_unobserved", after=1),
            _row(index=2, redirect=False, game="tn36", loop="selfparse", after=1),
        ],
    )

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path, outcomes=outcomes),
    )

    assert artifact["gate_check_summary"]["failed_check"] == "matched_cells"
    assert artifact["configuration_strata"]["matched_stratum_count"] == 0
    assert artifact["configuration_strata"]["unmatched_stratum_count"] == 3
    reasons = {row["reason"] for row in artifact["unmatched_cell_results"]}
    assert "no_matched_control_in_same_configuration" in reasons
    assert "control_without_redirect_in_same_configuration" in reasons


def test_scenario_6844_duplicates_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-6844-GATES-FAIL-CLOSED rejects duplicate exact identities."""

    redirect = _row(index=10, redirect=True)
    duplicate = deepcopy(redirect)
    duplicate["action_index"] = 11
    outcomes = _outcome_artifact(redirect_rows=[redirect, duplicate])

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path, outcomes=outcomes),
    )

    assert artifact["gate_check_summary"]["failed_check"] == "duplicate_exact_outcome_identities"
    assert artifact["exact_outcome_join_results"]["duplicate_outcome_ids"] == [
        redirect["outcome_id"]
    ]
    assert artifact["supervisor_effect_eligible_score"] == 0


def test_scenario_6844_prior_failures_are_diagnostics_not_credit(tmp_path: Path) -> None:
    """SCENARIO-ARC-6844-GATES-FAIL-CLOSED records partial verification failures."""

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path),
    )

    diagnostics = artifact["unmatched_cell_results"]
    assert any(
        row["source_artifact"] == "experiment_6682"
        and row["reason"] == "verification_failure"
        for row in diagnostics
    )
    assert any(
        row["source_artifact"] == "experiment_6524"
        and row["reason"] == "missing_outcome_bearing_receipts"
        for row in diagnostics
    )
    assert any(
        row["source_artifact"] == "experiment_6776"
        and row["reason"] == "resource_block"
        for row in diagnostics
    )


def test_scenario_6844_hashes_validator_lint_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-6844-HASHES-NO-SOLVE keeps hashes stable and writes JSON."""

    paths = _source_paths(tmp_path)
    artifact = exp.build_artifact(run_date="20260901", duration_s=0.25, source_paths=paths)

    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert all(
        row["row_sha256"].startswith("sha256:")
        for row in artifact["per_game_results"]
    )
    changed = deepcopy(artifact)
    changed["duration_s"] = 999.0
    assert exp.reproducibility_checksum(changed) == artifact["reproducibility_checksum"]
    assert discipline.duration_floor_s(exp.INFERENCE_SUBSTRATE) == 0.0001
    assert arc_lint.lint_artifact(
        tmp_path / "results/experiment_6844_supervisor_action_outcome_credit_audit.json",
        artifact,
    ) == []

    broken = deepcopy(artifact)
    broken["solve_claim"] = True
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "solve_claim must be false" in exp.validate_artifact(broken)

    monkeypatch.setattr(exp, "collect_default_source_paths", lambda _root: paths)
    output = tmp_path / "out.json"
    assert exp.main(["--date", "20260901", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["honest_verdict"].startswith("complete_")
    assert written["run_date"] == "20260901"

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced failure"])
    assert exp.main(["--date", "20260901", "--output", str(tmp_path / "bad.json")]) == 1


def test_scenario_6844_defensive_parsing_and_validation_branches(tmp_path: Path) -> None:
    """SCENARIO-ARC-6844-GATES-FAIL-CLOSED keeps malformed inputs non-positive."""

    assert exp._source_record("missing", tmp_path / "missing.json", tmp_path)["exists"] is False
    assert exp._load_json(b"{") == {}
    assert exp._load_json(b"[]") == {}
    assert "experiment_6681" in exp.collect_default_source_paths(tmp_path)
    assert exp._relative(Path("/not/under/root.json"), tmp_path).endswith("root.json")
    assert exp._outcome_score({"reward": {"present": True, "value": 3}}) == 3
    assert (
        exp._outcome_score(
            {
                "levels_completed_before": 0,
                "levels_completed_after": 0,
                "termination": {"state": "GAME_OVER"},
            }
        )
        == -1
    )
    assert exp._direction({"error": "boom", "outcome_status": "returned"}) == "regression"
    assert exp._lineage({"proposal_id": "p"})["proposal_id"] == "p"
    assert exp._temporal_order_ok({**_row(index=3, redirect=False), "decision_sequence": None})
    assert exp._action_validity({})["reason"] == "missing_proposed_action"
    assert (
        exp._action_validity({"proposed_action": {"kind": "RESET"}, "observation_before": {}})[
            "valid"
        ]
        is True
    )
    assert (
        exp._action_validity(
            {
                "proposed_action": {"kind": 6, "data": {"x": "bad", "y": 1}},
                "observation_before": {"available_actions": [6]},
            }
        )["reason"]
        == "action6_requires_integer_xy"
    )
    payload = {
        "canonical_path_receipt": {
            "policy": "canonical-policy",
            "live_metadata": {"episode_rows": ["bad", {"family": "tn36", "actions": 99}]},
        }
    }
    assert exp._episode_budget(payload, {"family": "tn36"}) == 99
    assert exp._episode_budget(payload, {"family": "missing"}) is None
    assert exp._policy(payload, {}) == "canonical-policy"
    assert exp._run({"family": "tn36", "attempt": 2, "episode_seed": 3}) == "tn36:2:3"
    assert exp.reduce_exact_outcome_rows({"redirect_outcome_rows": {}, "non_redirect_control_rows": []}) == []
    assert exp._rate_interval(0, 0)["lower"] is None
    assert exp._common_root(None) == exp.REPO_ROOT
    assert exp._common_root({"missing": tmp_path / "missing.json"}) == exp.REPO_ROOT

    inventory = _inventory()
    inventory["unmatched_cell_reasons"] = [
        {"stratum_identity": "inventory-only", "game": "tn36", "reasons": ["diagnostic"]}
    ]
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path, inventory=inventory),
    )
    assert any(row["matched_stratum"] == "inventory-only" for row in artifact["unmatched_cell_results"])

    def checked(mutator: Any, expected: str) -> None:
        broken = deepcopy(artifact)
        mutator(broken)
        if "reproducibility_checksum" in broken:
            broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
        assert expected in exp.validate_artifact(broken)

    checked(lambda data: data.pop("schema"), "required artifact fields are missing")
    checked(
        lambda data: data.update({"field_principles": {}}),
        "field principles do not cover every top-level field",
    )
    checked(lambda data: data.update({"schema": "bad"}), "schema mismatch")
    checked(lambda data: data.update({"inference_substrate": "bad"}), "inference substrate mismatch")
    checked(lambda data: data.update({"verifier_is_oracle": True}), "verifier_is_oracle must be false")
    checked(lambda data: data.update({"verdict_class": "surprise"}), "verdict class is outside the closed set")
    checked(lambda data: data.update({"honest_verdict": "blocked"}), "honest verdict lacks complete_ terminal prefix")
    broken_checksum = deepcopy(artifact)
    broken_checksum["status"] = "changed"
    assert "reproducibility checksum mismatch" in exp.validate_artifact(broken_checksum)
    checked(
        lambda data: data.update({"supervisor_causal_audit_complete_score": 0}),
        "causal audit complete score mismatch",
    )
    checked(
        lambda data: data.update({"supervisor_effect_eligible_score": 2}),
        "effect eligible score must be binary",
    )

    def blocked_without_failed_check(data: dict[str, Any]) -> None:
        data.update(
            {
                "status": "complete_blocked_supervisor_outcome_credit_audit",
                "verdict_class": "null",
                "supervisor_effect_eligible_score": 1,
                "gate_check_summary": {"passed": False},
            }
        )

    broken = deepcopy(artifact)
    blocked_without_failed_check(broken)
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    errors = exp.validate_artifact(broken)
    assert "blocked verdict_class mismatch" in errors
    assert "blocked artifact marked effect eligible" in errors
    assert "blocked artifact lacks failed check" in errors

    def complete_without_gate(data: dict[str, Any]) -> None:
        data.update(
            {
                "status": "complete_supervisor_outcome_credit_audit",
                "verdict_class": "blocked",
                "supervisor_effect_eligible_score": 0,
                "gate_check_summary": {"passed": False},
            }
        )

    broken = deepcopy(artifact)
    complete_without_gate(broken)
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    errors = exp.validate_artifact(broken)
    assert "complete audit verdict_class mismatch" in errors
    assert "complete artifact lacks effect eligibility" in errors
    assert "complete artifact gate summary mismatch" in errors

    checked(lambda data: data.update({"status": "unknown"}), "status mismatch")
    checked(lambda data: data.update({"per_game_results": []}), "per_game_results are empty")
    checked(
        lambda data: data["per_game_results"][0].update({"row_sha256": "bad"}),
        "per_game row missing hash",
    )
