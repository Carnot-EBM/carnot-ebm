"""Tests for Exp 4028 .372 Deep-Think pivot capstone.

Spec refs: REQ-CAPSTONE-4028, SCENARIO-CAPSTONE-4028.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v372_4028 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_payloads() -> dict[int, JsonDict]:
    return {
        4019: {
            "honest_verdict": "complete: archive_opened_v372",
            "active_milestone_confirmed": True,
            "milestone_371_closestate": {
                "arc3": {"total_games_solved": 5, "total_levels_solved": 5},
                "headline": "prior milestone closed at five ARC-3 games",
            },
        },
        4020: {
            "honest_verdict": "complete: goal_predicate_induced_heldout_precision_1.000",
            "game": "r11l",
            "goal_predicate_heldout_precision": 1.0,
            "heldout_recall": 1.0,
            "n_levelup_transitions": 3,
        },
        4021: {
            "honest_verdict": "complete: search_layer_solved_r11l_L4_real_env_confirmed",
            "game": "r11l",
            "new_levels_solved_this_task": 2,
            "wall_was_search_not_representation": True,
            "search_advanced_past_single_step_stall": True,
            "search_found_plan": True,
            "real_env_confirmed": True,
            "nodes_expanded": 9,
            "levels_completed_after": 5,
            "representation_vs_search_diagnosis": "fixture search solved a verified-model planning wall",
        },
        4022: {
            "honest_verdict": "complete: A_scale_exp4012_lift",
            "branch_taken": "A_scale",
            "decentralization_next_step": "run bounded same-pool scaling confirmation",
            "local_support_diagnostic": "latent_support_present",
        },
        4023: {
            "honest_verdict": "complete: agreement_selector_retired_confidence_label_only",
            "retired_r_and_d_line": "smart_selector_agreement_precision_confirmation",
            "agreement_role_after_retirement": "confidence_label_only",
            "agreement_is_precision_selector": False,
            "no_precision_confirmation_v4_proposed": True,
            "safety_gate_kept": True,
            "registry_updated": True,
            "retire_if_same_verdict_triggered": True,
        },
        4024: {
            "honest_verdict": "success: fifth_game_solved_cd82-fb555c5d_at_action_5",
            "prior_total_games_solved": 5,
            "total_games_solved": 7,
            "game_solved": True,
            "target_game": "cd82-fb555c5d",
            "real_env_confirmed": True,
            "candidate_baseline_actions": 55,
            "first_solve_at_action": 5,
        },
        4025: {
            "honest_verdict": "success: arcmemo_v5_transfer_10to7_actions",
            "solve_transfer_win": True,
            "actions_cold": 10,
            "actions_seeded": 7,
            "induction_calls_cold": 3,
            "induction_calls_seeded": 2,
        },
        4026: {
            "honest_verdict": "success: verifier_parity_wallclock_3x_judge_over_verifier",
            "accuracy_parity": True,
            "accuracy_gap": 0.01,
            "wallclock_seconds_ratio_judge_over_verifier": 3.0,
            "token_ratio_judge_over_verifier": 11.0,
            "verifier_gold_rate": 0.5,
            "judge_gold_rate": 0.49,
            "flagged_adversarial": False,
        },
        4027: {
            "honest_verdict": "complete: hardware_continuity_recorded",
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
        },
    }


def _write_artifacts(root: Path, payloads: dict[int, JsonDict]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for experiment_id, payload in payloads.items():
        path = root / "results" / f"experiment_{experiment_id}_fixture.json"
        _write_json(path, payload)
        paths[experiment_id] = path
    return paths


def _summary_statuses(ids: list[int] | tuple[int, ...]) -> dict[int, JsonDict]:
    return {
        experiment_id: {"returncode": 0, "stdout": f"summarized {experiment_id}", "stderr": ""}
        for experiment_id in ids
    }


def test_req_capstone_4028_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4028: OpenSpec declares the .372 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4028" in spec
    assert "SCENARIO-CAPSTONE-4028" in spec
    assert "pivot_central_bet_advanced" in spec
    assert "new_levels_this_milestone" in spec


def test_scenario_capstone_4028_current_artifacts_answer_deep_think_pivot() -> None:
    """SCENARIO-CAPSTONE-4028: current landed artifacts produce the .372 headline."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses={
            4019: {"returncode": 0, "stdout": "summary 4019", "stderr": ""},
            4020: {"returncode": 0, "stdout": "summary 4020", "stderr": ""},
            4021: {"returncode": 0, "stdout": "summary 4021", "stderr": ""},
            4022: {"returncode": 2, "stdout": "summary 4022", "stderr": ""},
            4023: {"returncode": 0, "stdout": "summary 4023", "stderr": ""},
            4024: {"returncode": 0, "stdout": "summary 4024", "stderr": ""},
            4025: {"returncode": 0, "stdout": "summary 4025", "stderr": ""},
            4026: {"returncode": 0, "stdout": "summary 4026", "stderr": ""},
            4027: {"returncode": 0, "stdout": "summary 4027", "stderr": ""},
        },
        started_s=10.0,
        now_s=12.5,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["pivot_central_bet_advanced"] is True
    assert artifact["new_levels_this_milestone"] == 1
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["planning_result"] == {
        "goal_predicate_game": "r11l",
        "goal_predicate_heldout_precision": 1.0,
        "new_levels_via_search": 1,
        "search_found_plan": True,
        "search_real_env_confirmed": True,
        "search_advanced_past_single_step_stall": True,
        "wall_was_search_not_representation": True,
        "levels_completed_after": 4,
        "nodes_expanded": 3,
    }
    assert artifact["search_vs_representation_diagnosis"].startswith("search found")
    assert artifact["decentralization_branch_taken"] == "skipped_flagged_exp4022"
    assert artifact["decentralization_result"]["status"] == "skipped_flagged_artifact"
    assert artifact["selection_retirement"]["agreement_role_after_retirement"] == "confidence_label_only"
    assert artifact["selection_retirement"]["safety_gate_kept"] is True
    assert artifact["accuracy_memory_efficiency_deltas"]["accuracy"] == {
        "prior_total_games_solved": 5,
        "total_games_solved": 6,
        "games_solved_delta": 1,
        "game_solved": True,
        "target_game": "cd82-fb555c5d",
        "real_env_confirmed": True,
    }
    assert artifact["accuracy_memory_efficiency_deltas"]["memory"]["action_savings"] == 50
    assert artifact["accuracy_memory_efficiency_deltas"]["efficiency"]["wallclock_seconds_ratio_judge_over_verifier"] == 95.2564
    assert artifact["duration_s"] == 2.5

    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 4022,
            "path": "results/experiment_4022_decentralization_gated.json",
            "reason": "flagged_adversarial:true",
        }
    ]
    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert 4022 not in cited
    assert set(cited) == {4019, 4020, 4021, 4023, 4024, 4025, 4026, 4027}
    assert cited[4021]["fields_imported"] == [
        "new_levels_solved_this_task",
        "wall_was_search_not_representation",
        "search_advanced_past_single_step_stall",
        "search_found_plan",
        "real_env_confirmed",
        "nodes_expanded",
        "levels_completed_after",
        "representation_vs_search_diagnosis",
    ]
    assert cited[4021]["sha256"] == hashlib.sha256(
        Path("results/experiment_4021_heuristic_search_over_verified_wm.json").read_bytes(),
    ).hexdigest()
    assert artifact["upstream_artifact_state"]["4022"]["included"] is False
    assert artifact["upstream_artifact_state"]["4022"]["flagged_adversarial"] is True
    mod.validate_artifact(artifact)


def test_req_capstone_4028_clean_upstreams_import_all_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4028: clean upstream metrics drive the planning and delta report."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["pivot_central_bet_advanced"] is True
    assert artifact["new_levels_this_milestone"] == 2
    assert artifact["planning_result"]["new_levels_via_search"] == 2
    assert artifact["planning_result"]["goal_predicate_heldout_precision"] == 1.0
    assert artifact["search_vs_representation_diagnosis"] == "fixture search solved a verified-model planning wall"
    assert artifact["decentralization_branch_taken"] == "A_scale"
    assert artifact["decentralization_result"]["status"] == "included"
    assert artifact["decentralization_result"]["local_support_diagnostic"] == "latent_support_present"
    assert artifact["selection_retirement"] == {
        "retired_r_and_d_line": "smart_selector_agreement_precision_confirmation",
        "agreement_role_after_retirement": "confidence_label_only",
        "agreement_is_precision_selector": False,
        "no_precision_confirmation_v4_proposed": True,
        "safety_gate_kept": True,
        "registry_updated": True,
        "retire_if_same_verdict_triggered": True,
    }
    assert artifact["accuracy_memory_efficiency_deltas"]["accuracy"]["games_solved_delta"] == 2
    assert artifact["accuracy_memory_efficiency_deltas"]["memory"] == {
        "solve_transfer_win": True,
        "actions_cold": 10,
        "actions_seeded": 7,
        "action_savings": 3,
        "induction_calls_cold": 3,
        "induction_calls_seeded": 2,
        "induction_call_savings": 1,
    }
    assert artifact["accuracy_memory_efficiency_deltas"]["efficiency"] == {
        "accuracy_parity": True,
        "accuracy_gap": 0.01,
        "wallclock_seconds_ratio_judge_over_verifier": 3.0,
        "token_ratio_judge_over_verifier": 11.0,
        "verifier_gold_rate": 0.5,
        "judge_gold_rate": 0.49,
    }
    assert 4022 in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}


def test_req_capstone_4028_flagged_missing_or_blocked_artifacts_count_false(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4028: flagged, missing, and blocked upstreams cannot satisfy gates."""

    payloads = _artifact_payloads()
    payloads[4021] = {
        **payloads[4021],
        "flagged_adversarial": True,
        "new_levels_solved_this_task": 99,
    }
    payloads[4022] = {
        **payloads[4022],
        "flagged_adversarial": True,
    }
    payloads[4024] = {
        "honest_verdict": "blocked_arc_offline_env_unavailable",
        "prior_total_games_solved": 5,
        "total_games_solved": 6,
        "game_solved": True,
        "real_env_confirmed": True,
    }
    payloads[4026] = {
        **payloads[4026],
        "flagged_adversarial": True,
    }
    payloads.pop(4025)
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["pivot_central_bet_advanced"] is False
    assert artifact["new_levels_this_milestone"] == 0
    assert artifact["planning_result"]["new_levels_via_search"] == 0
    assert artifact["decentralization_branch_taken"] == "skipped_flagged_exp4022"
    assert artifact["accuracy_memory_efficiency_deltas"]["accuracy"]["games_solved_delta"] == 0
    assert artifact["accuracy_memory_efficiency_deltas"]["memory"]["action_savings"] == 0
    assert artifact["accuracy_memory_efficiency_deltas"]["efficiency"]["accuracy_parity"] is False
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 4025}]
    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [4021, 4022, 4026]
    assert {4021, 4022, 4026}.isdisjoint({row["experiment_id"] for row in artifact["cited_upstream_artifacts"]})


def test_req_capstone_4028_write_artifact_and_validate_rejects_regressions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4028: artifact writing validates the required bare fields."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(paths)),
        started_s=1.0,
        now_s=1.25,
    )

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema"] == "carnot.capstone_v372_4028.v1"
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["pivot_central_bet_advanced"] = "true"
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["new_levels_this_milestone"] = True
    with pytest.raises(ValueError, match="bare int"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["inference_substrate"] = 4028
    with pytest.raises(ValueError, match="string"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["planning_result"] = []
    with pytest.raises(ValueError, match="planning_result"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["cited_upstream_artifacts"] = [{"experiment_id": 4021, "fields_imported": []}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["cited_upstream_artifacts"] = [{"experiment_id": 4021, "fields_imported": "not-list", "sha256": "f" * 64}]
    with pytest.raises(ValueError, match="fields_imported"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["flagged_artifacts_skipped"] = {}
    with pytest.raises(ValueError, match="list"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["reproducibility_checksum"] = "not-sha"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)


def test_req_capstone_4028_helpers_and_summary_runner_use_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4028: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / "results" / "experiment_4021_fixture.json"
    _write_json(path, _artifact_payloads()[4021])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(command, 0, stdout="summary", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    status = mod.run_summarize_artifact(tmp_path, path)

    assert status == {"returncode": 0, "stdout": "summary", "stderr": ""}
    assert calls == [[str(mod.PYTHON_BIN), "scripts/summarize_artifact.py", str(path)]]

    monkeypatch.setattr(
        mod,
        "run_summarize_artifact",
        lambda root, artifact_path: {
            "returncode": 0,
            "stdout": f"summary for {artifact_path.name}",
            "stderr": "",
        },
    )
    statuses = mod.summarize_existing_artifacts(tmp_path, {4021: path, 4025: None}, supplied=None)
    assert statuses == {
        4021: {
            "returncode": 0,
            "stdout": "summary for experiment_4021_fixture.json",
            "stderr": "",
        }
    }
    assert mod.summarize_existing_artifacts(tmp_path, {4021: path}, supplied={4021: {"returncode": 2}}) == {
        4021: {"returncode": 2}
    }
    assert mod.float_metric({"x": True}, "x") == 0.0
    assert mod.float_metric({"x": 2}, "x") == 2.0
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.str_metric({"x": 3}, "x") == ""
    assert mod.nested_int({"a": {"b": 4}}, ("a", "b")) == 4
    assert mod.nested_int({"a": True}, ("a", "b")) == 0
    assert mod.decentralization_result(None, False) == {"status": "missing_or_blocked", "branch_taken": ""}
    assert mod.is_sha256("f" * 64) is True
    assert mod.is_sha256("z" * 64) is False


def test_scenario_capstone_4028_script_wrapper_exists() -> None:
    """SCENARIO-CAPSTONE-4028: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_4028_capstone_v372.py")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "capstone_v372_4028" in text
    assert "write_artifact" in text
