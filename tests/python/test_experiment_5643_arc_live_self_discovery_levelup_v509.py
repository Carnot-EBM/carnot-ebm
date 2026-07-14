"""Tests for Exp5643 ARC live self-discovery level-up attempt.

Spec refs: REQ-ARC-FCP-5643,
SCENARIO-ARC-FCP-5643-PRECHECK-EXCLUDES-TRANSITION-FAILURES,
SCENARIO-ARC-FCP-5643-EXECUTABLE-POLICY-ADVISORY-NOT-GATING,
SCENARIO-ARC-FCP-5643-METHODOLOGY-AND-REPRODUCTION-GATE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5643_arc_live_self_discovery_levelup_v509 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH

sys.path.insert(0, str(REPO / "scripts"))
import adversarial_verify as av  # noqa: E402


def _registry(*, total: int = 177) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "lf52", "reproducibility": "reproduced", "levels_reproduced": 6},
            {"game": "sk48", "reproducibility": "reproduced", "levels_reproduced": 7},
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 8},
            {"game": "done", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _public_envs() -> dict[str, dict[str, Any]]:
    return {
        "lf52": {"game_id": "lf52-public", "baseline_actions": [1] * 10, "tags": ["click"]},
        "sk48": {"game_id": "sk48-public", "baseline_actions": [1] * 8, "tags": ["keyboard_click"]},
        "bp35": {"game_id": "bp35-public", "baseline_actions": [1] * 9, "tags": ["keyboard_click"]},
        "done": {"game_id": "done-public", "baseline_actions": [1] * 2, "tags": ["click"]},
    }


def _transition_receipt() -> dict[str, Any]:
    return {
        "experiment": "experiment_5636_transition_v509",
        "retired_scopes": [
            {
                "key": "arc_live_level_credit_v508",
                "reason": "The bounded live attempt executed but banked no new reproducible level.",
                "source_artifacts": [
                    "results/experiment_5632_arc_live_self_discovery_levelup_v508.json"
                ],
                "evidence": {
                    "selected_game": "lf52",
                    "selected_level": "L7",
                    "registry_delta": 0,
                    "offline_reproduced": False,
                },
            }
        ],
    }


def _precheck() -> dict[str, Any]:
    return mod.registry_precheck(
        registry=_registry(),
        public_envs=_public_envs(),
        arc_loop_depths={"lf52": 2, "sk48": 2, "bp35": 2, "done": 2},
        transition_receipt=_transition_receipt(),
    )


def _target() -> dict[str, Any]:
    return mod.select_target_from_precheck(_precheck())


def _policy_source() -> dict[str, Any]:
    return mod.policy_source_from_exp5642(
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "executable model upstream readiness failed",
        }
    )


def _null_attempt() -> dict[str, Any]:
    rows = [
        {"step": 0, "kind": "RESET", "action": None, "data": None, "level_before": None, "level_after": 0},
        {"step": 1, "kind": "ACTION", "action": 6, "data": {"x": 10, "y": 10}, "level_before": 0, "level_after": 0},
        {"step": 2, "kind": "ACTION", "action": 1, "data": None, "level_before": 0, "level_after": 0},
    ]
    checksum = mod.action_trace_sha256(rows)
    return {
        "live_attempt_executed": True,
        "action_budget": mod.ACTION_BUDGET,
        "action_rows": rows,
        "observations": [{"step": 0, "event": "reset", "level": 0}],
        "level_counter_changes": [],
        "runtime_reverse_engineering": {
            "source": "runtime_observations_actions_state_transitions_only",
            "observations_recorded": 2,
            "source_files_read": False,
            "per_game_adapter_used": False,
            "offline_ground_truth_bfs_used": False,
        },
        "max_level_reached": 0,
        "post_levels_reproduced": 6,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "reproduction_gate": {
            "game": "lf52",
            "claimed_level": 0,
            "reached_level": 6,
            "reproduced": False,
            "mode": "standard_reproduction_gate_not_run_no_new_target_candidate",
        },
        "solution_labels": [],
        "action_trace_sha256": checksum,
        "trace_replay_checksum": checksum,
        "llm_invoked": False,
        "model_specs": [],
        "source_files_read": False,
        "per_game_adapter_used": False,
        "offline_bfs_used": False,
        "outer_loop_re_used": False,
        "terminal_reason": "bounded_budget_no_target_level_reproduction",
    }


def _success_attempt() -> dict[str, Any]:
    rows = [
        {"step": 0, "kind": "RESET", "action": None, "data": None, "level_before": None, "level_after": 0},
        {"step": 1, "kind": "ACTION", "action": 6, "data": {"x": 20, "y": 20}, "level_before": 6, "level_after": 9},
    ]
    checksum = mod.action_trace_sha256(rows)
    return {
        **_null_attempt(),
        "action_rows": rows,
        "level_counter_changes": [{"step": 1, "level_before": 6, "level_after": 9}],
        "max_level_reached": 9,
        "post_levels_reproduced": 9,
        "offline_reproduced": True,
        "reproduced_levels": 3,
        "reproduction_gate": {
            "game": "lf52",
            "claimed_level": 9,
            "reached_level": 9,
            "reproduced": True,
            "mode": "generic_live_path_clean_state_reproduction",
        },
        "solution_labels": ['{"action":6,"data":{"x":20,"y":20}}'],
        "action_trace_sha256": checksum,
        "trace_replay_checksum": checksum,
        "terminal_reason": "target_level_reached_live",
    }


def test_req_arc_fcp_5643_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5643: OpenSpec anchors the V509 live level-up receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5643") :]

    for marker in (
        mod.RESULT_RELATIVE_PATH,
        "SCENARIO-ARC-FCP-5643-PRECHECK-EXCLUDES-TRANSITION-FAILURES",
        "SCENARIO-ARC-FCP-5643-EXECUTABLE-POLICY-ADVISORY-NOT-GATING",
        "SCENARIO-ARC-FCP-5643-METHODOLOGY-AND-REPRODUCTION-GATE",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_scenario_5643_precheck_excludes_transition_failures_and_rotates_to_lf52_l8() -> None:
    """SCENARIO-ARC-FCP-5643-PRECHECK-EXCLUDES-TRANSITION-FAILURES."""

    failed = mod.recent_failed_targets_from_transition_receipt(_transition_receipt())
    precheck = _precheck()
    target = mod.select_target_from_precheck(precheck)
    lf52 = next(row for row in precheck["candidate_rows"] if row["game"] == "lf52")
    sk48 = next(row for row in precheck["candidate_rows"] if row["game"] == "sk48")
    bp35 = next(row for row in precheck["candidate_rows"] if row["game"] == "bp35")
    done = next(row for row in precheck["candidate_rows"] if row["game"] == "done")

    assert failed == [
        {
            "game": "lf52",
            "level": 7,
            "level_label": "L7",
            "reason": "transition_receipt_failed_arc_live_level_credit_v508",
            "source": "results/experiment_5636_transition_v509.json",
        }
    ]
    assert precheck["ok"] is True
    assert "recent_failed_target" in lf52["closed_level_reasons"]["7"]
    assert "explicit_recent_unbanked_attempt" in sk48["closed_level_reasons"]["8"]
    assert "explicit_recent_unbanked_attempt" in bp35["closed_level_reasons"]["9"]
    assert "no_authenticated_headroom" in done["exclude_reasons"]
    assert target["selected_game"] == "lf52"
    assert target["selected_level"] == "L8"
    assert target["prior_levels_reproduced"] == 6
    assert target["target_level"] == 8
    assert target["closed_intermediate_levels"] == [7]
    assert target["selected_target_was_unreproduced"] is True
    assert target["target_selection_hash"].startswith("sha256:")


def test_scenario_5643_exp5642_policy_is_advisory_not_gating() -> None:
    """SCENARIO-ARC-FCP-5643-EXECUTABLE-POLICY-ADVISORY-NOT-GATING."""

    blocked = _policy_source()
    promoted = mod.policy_source_from_exp5642(
        {
            "status": "complete",
            "honest_verdict": "complete: executable model live ab promoted",
            "live_executable_model_ready_score": 1.0,
            "unsafe_model_accept_count": 0,
            "known_level_regression_count": 0,
            "treatment_policy": {"name": "executable_model_policy_v1"},
        }
    )
    unsafe = mod.policy_source_from_exp5642(
        {
            "status": "complete",
            "live_executable_model_ready_score": 1.0,
            "unsafe_model_accept_count": 1,
            "known_level_regression_count": 0,
            "treatment_policy": {"name": "executable_model_policy_v1"},
        }
    )

    assert blocked["attempt_gated_by_exp5642"] is False
    assert blocked["policy_name"] == "unchanged_no_new_llm_e3_baseline"
    assert blocked["baseline_unchanged"] is True
    assert promoted["policy_name"] == "promoted_exp5642_executable_model_policy"
    assert promoted["baseline_unchanged"] is False
    assert unsafe["policy_name"] == "unchanged_no_new_llm_e3_baseline"


def test_scenario_5643_methodology_and_reproduction_gate_bank_at_most_one_level(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5643-METHODOLOGY-AND-REPRODUCTION-GATE."""

    null_artifact = mod.build_artifact(
        registry_precheck_receipt=_precheck(),
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_null_attempt(),
        live_trace_path="results/null-5643.json",
        duration_s=0.03,
        tests_run=["unit"],
    )
    success = mod.build_artifact(
        registry_precheck_receipt=_precheck(),
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_success_attempt(),
        live_trace_path="results/success-5643.json",
        duration_s=0.03,
        tests_run=["unit"],
    )

    mod.validate_artifact(null_artifact)
    mod.validate_artifact(success)
    assert null_artifact["methodology_receipt"]["source_read"] is False
    assert null_artifact["methodology_receipt"]["model_call_limit"] == 0
    assert null_artifact["model_specs"] == []
    assert null_artifact["random_seeds"] == [5643]
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["reproduced_levels"] == 0
    assert null_artifact["registry_count_before"] == 177
    assert null_artifact["registry_count_after"] == 177
    assert null_artifact["registry_delta"] == 0
    assert null_artifact["honest_verdict"].startswith("complete:")
    assert success["offline_reproduced"] is True
    assert success["new_reproducible_levels"] == [{"game": "lf52", "level": 8}]
    assert success["reproduced_levels"] == 1
    assert success["level_reached"] == 9
    assert success["registry_count_after"] == 178
    assert success["registry_delta"] == 1

    trace = mod.build_live_trace(
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_null_attempt(),
        artifact=null_artifact,
    )
    trace_path = tmp_path / "trace.json"
    mod.write_json(trace_path, trace)
    loaded = json.loads(trace_path.read_text(encoding="utf-8"))
    assert loaded["action_trace_sha256"] == null_artifact["action_trace_sha256"]
    assert loaded["executed_actions"] == _null_attempt()["action_rows"]


def test_req_arc_fcp_5643_validation_rejects_missing_fields_and_drift() -> None:
    """REQ-ARC-FCP-5643: malformed artifacts fail closed before being written."""

    artifact = mod.build_artifact(
        registry_precheck_receipt=_precheck(),
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_null_attempt(),
        live_trace_path="results/null-5643.json",
        duration_s=0.03,
        tests_run=["unit"],
    )
    missing = dict(artifact)
    missing.pop("methodology_receipt")
    drifted = dict(artifact)
    drifted["registry_count_after"] = 999
    bad_model_specs = dict(artifact)
    bad_model_specs["model_specs"] = ["legacy-small-model"]
    bad_source = dict(artifact)
    bad_source["source_read"] = True

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        mod.validate_artifact(drifted)
    with pytest.raises(ValueError, match="no-LLM attempts require model_specs=\\[\\]"):
        mod.validate_artifact(bad_model_specs)
    with pytest.raises(ValueError, match="source_read must be false"):
        mod.validate_artifact(bad_source)


def test_req_arc_fcp_5643_adversarial_verify_accepts_no_llm_live_substrate(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5643: no-LLM live environment interaction is not treated as a
    missing model-spec or too-short live-GGUF invocation."""

    artifact = mod.build_artifact(
        registry_precheck_receipt=_precheck(),
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_null_attempt(),
        live_trace_path="results/null-5643.json",
        duration_s=0.03,
        tests_run=["unit"],
    )
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    report = av.verify_artifact(path)
    flag_kinds = {flag["kind"] for flag in report["flags"]}
    assert "DURATION_TOO_SHORT" not in flag_kinds
    assert "METHODOLOGY_MISSING" not in flag_kinds


def test_req_arc_fcp_5643_repository_artifact_has_required_schema() -> None:
    """REQ-ARC-FCP-5643: checked-in Exp5643 artifact is the stable live-attempt receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["solve_provenance"] == mod.SOLVE_PROVENANCE
    assert artifact["source_read"] is False
    assert artifact["game_adapter_used"] is False
    assert artifact["outer_loop_re_used"] is False
    assert artifact["model_specs"] == []
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["selected_game"]
    assert artifact["selected_level"]
    assert artifact["registry_delta"] in (0, 1)
    for field in mod.REQUIRED_FIELDS:
        assert field in artifact
