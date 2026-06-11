"""Tests for Exp 4053 .374 decision-grade capstone.

Spec refs: REQ-CAPSTONE-4053, SCENARIO-CAPSTONE-4053.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v374_4053 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_statuses(
    experiment_ids: tuple[int, ...] = mod.UPSTREAM_IDS,
    *,
    returncodes: dict[int, int] | None = None,
) -> dict[int, JsonDict]:
    overrides = returncodes or {}
    return {
        experiment_id: {
            "returncode": overrides.get(experiment_id, 0),
            "stdout": f"summarized {experiment_id}",
            "stderr": "",
        }
        for experiment_id in experiment_ids
    }


def _fixture_payloads() -> dict[int, JsonDict]:
    return {
        4042: {
            "honest_verdict": "success: archived_v373_v374_active",
            "milestone_373_closestate": {"accuracy": {"total_games_solved": 7}},
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        4043: {
            "honest_verdict": "complete: sota_ingestion_offarc_power_and_closed_loop_mapped",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        4044: {
            "honest_verdict": "success: offarc_power_runner_built_smoked_launched",
            "inference_substrate": "live_llm_inference",
        },
        4045: {
            "honest_verdict": "complete: offarc_power_full_n_demofit_ci_excludes_zero",
            "n_tasks": 160,
            "powered_task_floor": 160,
            "raw_artifact_present": True,
            "partial_reason": None,
            "demofit_delta_pp": 4.5,
            "demofit_bootstrap_ci95": [1.1, 8.2],
            "demofit_ci_excludes_zero": True,
            "best_arm": "armB_demofit",
            "best_arm_delta_pp": 4.5,
            "best_arm_ci95": [1.1, 8.2],
            "best_arm_ci_excludes_zero": True,
            "oracle_passrate": 0.72,
            "oracle_headroom": True,
        },
        4046: {
            "honest_verdict": "complete: closed_loop_solved_vc33_L1_real_env_confirmed",
            "game": "vc33",
            "closed_loop_broke_wall": True,
            "per_step_wm_real_divergence_rate": 0.0,
            "divergence_gate_fired_count": 0,
            "real_env_confirmed": True,
            "new_levels_solved_this_task": 1,
            "levels_completed_after": 1,
            "bottleneck": "",
        },
        4047: {
            "honest_verdict": "success: decentralization_moe_base_runner_launched",
            "inference_substrate": "live_llm_inference",
        },
        4048: {
            "honest_verdict": "complete: decentralization_moe_base_cov_1_latent_distill_viable",
            "raw_complete": True,
            "n_tasks_scored": 31,
            "moe_base_demo_perfect_coverage": 1.0,
            "coverage_delta_vs_12b": 0.7419,
            "bootstrap_ci95": [0.7419, 0.7419],
            "local_support_diagnosis": "latent",
            "gated_pass_at_2": 0.7,
            "oracle_coverage": 0.6129,
        },
        4049: {
            "honest_verdict": "success: eighth_game_solved",
            "game_solved": True,
            "real_env_confirmed": True,
            "prior_total_games_solved": 7,
            "total_games_solved": 8,
            "target_game": "fixture-game",
            "candidate_baseline_actions": 18,
            "first_solve_at_action": 9,
            "exploration_actions_used": 2,
        },
        4050: {
            "honest_verdict": "complete: arcmemo_v7_cross_game_transfer_win",
            "cross_game_transfer_win": True,
            "actions_cold": 18,
            "actions_within_game_v6": 11,
            "actions_cross_game_v7": 8,
            "induction_calls_cold": 1,
            "induction_calls_within_game_v6": 1,
            "induction_calls_cross_game_v7": 0,
            "n_prior_fragments": 7,
            "n_named_abstractions": 1,
            "n_reused_abstractions": 1,
            "transfer_assessment": "fixture_win",
        },
        4051: {
            "honest_verdict": "complete: hygiene_done",
            "offline_reeval_bitexact": True,
            "registry_updated": True,
            "gaps_updated": True,
        },
        4052: {
            "honest_verdict": "complete: hardware_continuity",
            "kv260_overlay_loaded": True,
            "kv260_latency_step_taken": True,
            "kv260_latency_median_ms": 0.01,
            "kv260_latency_batch_ms": 0.5,
            "speedup_claim_made": False,
            "fabric_acceleration_claimed": False,
            "per_board_reachability": {"kv260": True, "gatemate": True, "polarfire": True},
            "per_board_terminal_state": {
                "kv260": "latency",
                "gatemate": "reachable",
                "polarfire": "reachable",
            },
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4053_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4053: OpenSpec declares the .374 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4053" in spec
    assert "SCENARIO-CAPSTONE-4053" in spec
    assert "verifier_transferred_off_arc_significantly" in spec
    assert "search_layer_salvageable_closed_loop" in spec
    assert "decentralization_diagnosis" in spec


def test_scenario_capstone_4053_current_artifacts_emit_honest_headline() -> None:
    """SCENARIO-CAPSTONE-4053: landed artifacts produce negative and retired outcomes."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses=_summary_statuses(returncodes={4044: 2, 4047: 2}),
        started_s=10.0,
        now_s=12.0,
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith("complete:")
    assert "not_decision_grade" in artifact["honest_verdict"]
    assert artifact["verifier_transferred_off_arc_significantly"] is False
    assert artifact["g1_off_arc_transfer"]["n_tasks"] == 22
    assert artifact["g1_off_arc_transfer"]["powered_task_floor"] == 160
    assert artifact["g1_off_arc_transfer"]["full_power_reached"] is False
    assert artifact["g1_off_arc_transfer"]["demofit_delta_pp"] == 0.0
    assert artifact["g1_off_arc_transfer"]["demofit_bootstrap_ci95"] == [0.0, 0.0]
    assert artifact["g1_off_arc_transfer"]["demofit_ci_excludes_zero"] is False
    assert artifact["g1_off_arc_transfer"]["best_arm"] == "armC_symbolic"

    assert artifact["search_layer_salvageable_closed_loop"] is False
    assert artifact["g2_closed_loop_grounding"]["closed_loop_broke_wall"] is False
    assert artifact["g2_closed_loop_grounding"]["per_step_wm_real_divergence_rate"] == 0.207031
    assert artifact["g2_closed_loop_grounding"]["real_env_confirmed"] is False
    assert artifact["g2_closed_loop_grounding"]["bottleneck"] == "wm_real_divergence_gate_fired"

    assert artifact["decentralization_diagnosis"] == "retired_non_measurement"
    assert artifact["g3_decentralization_moe_base"]["baseline_12b_coverage"] == 0.2581
    assert artifact["g3_decentralization_moe_base"]["moe_base_demo_perfect_coverage"] == 0.5
    assert artifact["g3_decentralization_moe_base"]["coverage_delta_vs_12b"] == 0.2419
    assert artifact["g3_decentralization_moe_base"]["n_tasks_scored"] == 6
    assert artifact["g3_decentralization_moe_base"]["raw_complete"] is False

    assert artifact["total_games_solved"] == 8
    assert artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["games_solved_delta"] == 1
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"]["cross_game_transfer_win"] is False
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"]["action_savings_vs_cold"] == 9
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"]["cross_game_extra_actions_vs_within_game_v6"] == 2
    assert artifact["accuracy_self_learning_hardware_deltas"]["hardware"]["kv260_overlay_loaded"] is True
    assert artifact["accuracy_self_learning_hardware_deltas"]["hardware"]["kv260_latency_step_taken"] is True

    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [4044, 4047]
    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert {4044, 4047}.isdisjoint(cited)
    assert set(cited) == {4042, 4043, 4045, 4046, 4048, 4049, 4050, 4051, 4052}
    assert cited[4045] == {
        "experiment_id": 4045,
        "sha256": hashlib.sha256(Path("results/experiment_4045_offarc_transfer_power.json").read_bytes()).hexdigest(),
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["verifier_transferred_off_arc_significantly"].startswith("BARE BOOL")


def test_req_capstone_4053_clean_fixture_can_record_decision_grade_success(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4053: clean upstream metrics can satisfy all three capstone axes."""

    payloads = _fixture_payloads()
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(), started_s=1.0, now_s=1.25)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: capstone_v374_decision_grade_yes")
    assert artifact["verifier_transferred_off_arc_significantly"] is True
    assert artifact["g1_off_arc_transfer"]["outcome"] == "significant_full_power"
    assert artifact["search_layer_salvageable_closed_loop"] is True
    assert artifact["g2_closed_loop_grounding"]["outcome"] == "closed_loop_broke_wall"
    assert artifact["decentralization_diagnosis"] == "latent"
    assert artifact["g3_decentralization_moe_base"]["outcome"] == "decision_grade_measurement"
    assert artifact["total_games_solved"] == 8
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"]["cross_game_extra_actions_vs_within_game_v6"] == -3
    assert artifact["decision_grade_measurements"] == {
        "G1": True,
        "G2": True,
        "G3": True,
        "all_three": True,
    }
    assert artifact["flagged_artifacts_skipped"] == []
    assert len(artifact["cited_upstream_artifacts"]) == len(mod.UPSTREAM_IDS)
    assert artifact["duration_s"] == 0.25


def test_req_capstone_4053_flagged_and_missing_inputs_cannot_satisfy_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4053: skipped or missing upstreams do not leak headline metrics."""

    payloads = _fixture_payloads()
    payloads[4045]["flagged_adversarial"] = True
    payloads[4046]["flagged_adversarial"] = True
    payloads[4048]["flagged_adversarial"] = True
    payloads.pop(4049)
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(payloads), returncodes={4045: 2, 4046: 2, 4048: 2}),
    )

    mod.validate_artifact(artifact)
    assert artifact["verifier_transferred_off_arc_significantly"] is False
    assert artifact["g1_off_arc_transfer"]["outcome"] == "skipped_flagged"
    assert artifact["search_layer_salvageable_closed_loop"] is False
    assert artifact["g2_closed_loop_grounding"]["outcome"] == "skipped_flagged"
    assert artifact["decentralization_diagnosis"] == "flagged_skipped"
    assert artifact["total_games_solved"] == 7
    assert artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["game_solved"] is False
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 4049}]
    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [4045, 4046, 4048]
    assert {4045, 4046, 4048}.isdisjoint({row["experiment_id"] for row in artifact["cited_upstream_artifacts"]})


def test_req_capstone_4053_write_artifact_and_validate_rejects_regressions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4053: artifact writing validates required schema fields."""

    _write_default_artifacts(tmp_path, _fixture_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.5,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert written["schema"] == "carnot.capstone_v374_4053.v1"
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["verifier_transferred_off_arc_significantly"] = "true"
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["search_layer_salvageable_closed_loop"] = 1
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["decentralization_diagnosis"] = "partial"
    with pytest.raises(ValueError, match="decentralization_diagnosis"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["total_games_solved"] = True
    with pytest.raises(ValueError, match="bare int"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="aggregation_from_upstream_artifacts"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["cited_upstream_artifacts"] = [{"experiment_id": 4045, "sha256": "not-sha"}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["flagged_artifacts_skipped"] = {}
    with pytest.raises(ValueError, match="list"):
        mod.validate_artifact(bad)


def test_req_capstone_4053_summary_runner_uses_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4053: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / mod.DEFAULT_UPSTREAM_PATHS[4045]
    _write_json(path, _fixture_payloads()[4045])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(command, 0, stdout="summary", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    assert mod.run_summarize_artifact(tmp_path, path) == {
        "returncode": 0,
        "stdout": "summary",
        "stderr": "",
    }
    assert calls == [[str(mod.PYTHON_BIN), "scripts/summarize_artifact.py", str(path)]]

    statuses = mod.summarize_existing_artifacts(tmp_path, {4045: path, 4049: None}, supplied=None)
    assert statuses == {4045: {"returncode": 0, "stdout": "summary", "stderr": ""}}
    assert mod.summarize_existing_artifacts(tmp_path, {4045: path}, supplied={4045: {"returncode": 2}}) == {
        4045: {"returncode": 2}
    }
    assert mod.list_float_metric({"ci": "not-a-list"}, "ci") == []
    assert mod.list_float_metric({"ci": [1, True, 2.5, "bad"]}, "ci") == [1.0, 2.5]
    assert mod.nested_int({"a": True}, ("a", "b")) == 0
    assert mod.off_arc_transfer_report(None, was_skipped=False)["outcome"] == "missing_or_blocked"
    assert (
        mod.off_arc_transfer_report(
            {
                "honest_verdict": "complete: full_n_no_headroom",
                "n_tasks": 160,
                "powered_task_floor": 160,
                "demofit_delta_pp": 0.0,
                "demofit_ci_excludes_zero": False,
                "oracle_headroom": False,
            },
            was_skipped=False,
        )["outcome"]
        == "ceiling_saturated_no_headroom"
    )
    assert (
        mod.off_arc_transfer_report(
            {
                "honest_verdict": "complete: full_n_not_significant",
                "n_tasks": 160,
                "powered_task_floor": 160,
                "demofit_delta_pp": 1.0,
                "demofit_ci_excludes_zero": False,
                "oracle_headroom": True,
            },
            was_skipped=False,
        )["outcome"]
        == "not_significant_full_power"
    )
    assert mod.closed_loop_grounding_report(None, was_skipped=False)["outcome"] == "missing_or_blocked"
    assert (
        mod.closed_loop_grounding_report(
            {
                "honest_verdict": "complete: closed_loop_no_break",
                "closed_loop_broke_wall": False,
                "real_env_confirmed": False,
                "per_step_wm_real_divergence_rate": 0.0,
                "divergence_gate_fired_count": 0,
            },
            was_skipped=False,
        )["outcome"]
        == "closed_loop_no_break"
    )
    assert mod.decentralization_moe_report(None, was_skipped=False)["outcome"] == "missing_or_blocked"
    assert (
        mod.decentralization_moe_report(
            {
                "honest_verdict": "complete: saturated_pool",
                "raw_complete": True,
                "n_tasks_scored": 31,
                "local_support_diagnosis": "uninformative",
            },
            was_skipped=False,
        )["outcome"]
        == "uninformative_measurement"
    )
    fallback = mod.decentralization_moe_report(
        {
            "honest_verdict": "complete: missing_explicit_diagnosis",
            "raw_complete": True,
            "n_tasks_scored": 31,
            "moe_base_demo_perfect_coverage": 0.9,
        },
        was_skipped=False,
    )
    assert fallback["outcome"] == "decision_grade_measurement"
    assert fallback["diagnosis"] == "latent"
