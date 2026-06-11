"""Tests for Exp 4041 .373 argument-measurement capstone.

Spec refs: REQ-CAPSTONE-4041, SCENARIO-CAPSTONE-4041.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v373_4041 as mod


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
        4029: {
            "honest_verdict": "success: archived_v372_v373_active",
            "milestone_372_closestate": {"arc3": {"total_games_solved": 6}},
        },
        4030: {"honest_verdict": "complete: sota_ingestion_offarc_and_search_mapped"},
        4031: {"honest_verdict": "success: offarc_transfer_runner_built_smoked_launched"},
        4032: {
            "honest_verdict": "success: offarc_exec_verifier_generalized_ci_excludes_zero",
            "n_tasks": 80,
            "delta_pp": 8.0,
            "bootstrap_ci95_pp": [2.0, 14.0],
            "ci_excludes_zero": True,
            "positive_control_passes": True,
            "three_outcome_verdict": "OUTCOME_1: transfer confirmed",
        },
        4033: {"honest_verdict": "complete: verifier_registered"},
        4034: {
            "honest_verdict": "complete: vc33_goal_predicate_induced",
            "game": "vc33",
            "goal_predicate_heldout_precision": 1.0,
        },
        4035: {
            "honest_verdict": "success: search_layer_generalized_vc33",
            "game": "vc33",
            "search_layer_generalizes": True,
            "heuristic_was_non_bespoke": True,
            "nodes_expanded": 42,
            "search_found_plan": True,
            "real_env_confirmed": True,
            "levels_completed_after": 1,
            "new_levels_solved_this_task": 1,
            "goal_predicate_heldout_precision": 1.0,
        },
        4036: {"honest_verdict": "success: stronger_base_runner_launched"},
        4037: {
            "honest_verdict": "success: decentralization_stronger_base_latent",
            "stronger_base_demo_perfect_coverage": 0.5,
            "coverage_delta_vs_12b": 0.2419,
            "gated_pass_at_2": 0.61,
            "local_support_diagnosis": "latent",
            "n_tasks_scored": 31,
            "local_seconds_per_task": 43.0,
        },
        4038: {
            "honest_verdict": "success: eighth_game_solved",
            "prior_total_games_solved": 7,
            "total_games_solved": 8,
            "game_solved": True,
            "target_game": "fixture-game",
            "real_env_confirmed": True,
            "candidate_baseline_actions": 33,
            "first_solve_at_action": 12,
        },
        4039: {
            "honest_verdict": "success: arcmemo_v6_library_transfer_33to11_actions",
            "solve_transfer_win": True,
            "actions_cold": 33,
            "actions_v5": 13,
            "actions_v6": 11,
            "induction_calls_cold": 2,
            "induction_calls_v5": 1,
            "induction_calls_v6": 0,
            "n_named_abstractions": 3,
        },
        4040: {
            "honest_verdict": "complete: hardware_continuity_fixture",
            "kv260_overlay_loaded": True,
            "kv260_latency_step_taken": True,
            "speedup_claim_made": False,
            "fabric_acceleration_claimed": False,
            "per_board_reachability": {"kv260": True, "gatemate": True, "polarfire": True},
            "per_board_terminal_state": {
                "kv260": "latency_samples_recorded",
                "gatemate": "reachable",
                "polarfire": "reachable",
            },
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4041_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4041: OpenSpec declares the .373 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4041" in spec
    assert "SCENARIO-CAPSTONE-4041" in spec
    assert "verifier_generalized_off_arc" in spec
    assert "search_layer_generalized" in spec
    assert "decentralization_diagnosis" in spec


def test_scenario_capstone_4041_current_artifacts_emit_honest_headline() -> None:
    """SCENARIO-CAPSTONE-4041: landed .373 artifacts become measurements, not inflated wins."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses=_summary_statuses(returncodes={4031: 2}),
        started_s=10.0,
        now_s=12.5,
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith("complete:")
    assert "arguments_measured" in artifact["honest_verdict"]
    assert artifact["verifier_generalized_off_arc"] is False
    assert artifact["g1_off_arc_transfer"]["outcome"] == "directional_underpowered_ci_touches_zero"
    assert artifact["g1_off_arc_transfer"]["delta_pp"] == 5.0
    assert artifact["g1_off_arc_transfer"]["bootstrap_ci95_pp"] == [0.0, 12.5]
    assert artifact["g1_off_arc_transfer"]["ci_excludes_zero"] is False
    assert artifact["g1_off_arc_transfer"]["positive_control_passes"] is True

    assert artifact["search_layer_generalized"] is False
    assert artifact["g2_search_layer_generalization"]["game"] == "vc33"
    assert artifact["g2_search_layer_generalization"]["heuristic_was_non_bespoke"] is True
    assert artifact["g2_search_layer_generalization"]["nodes_expanded"] == 169
    assert artifact["g2_search_layer_generalization"]["search_found_plan"] is True
    assert artifact["g2_search_layer_generalization"]["real_env_confirmed"] is False

    assert artifact["decentralization_diagnosis"] == "absent"
    assert artifact["g3_decentralization_scaling"]["baseline_12b_coverage"] == 0.2581
    assert artifact["g3_decentralization_scaling"]["stronger_base_demo_perfect_coverage"] == 0.0
    assert artifact["g3_decentralization_scaling"]["coverage_delta_vs_12b"] == -0.2581

    assert artifact["total_games_solved"] == 7
    assert artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["games_solved_delta"] == 1
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"]["solve_transfer_win"] is True
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"]["action_savings_vs_cold"] == 41
    assert artifact["accuracy_self_learning_hardware_deltas"]["hardware"]["kv260_overlay_loaded"] is True
    assert artifact["accuracy_self_learning_hardware_deltas"]["hardware"]["kv260_latency_step_taken"] is False

    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 4031,
            "path": "results/experiment_4031_offarc_transfer_build.json",
            "reason": "flagged_adversarial:true",
        }
    ]
    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert 4031 not in cited
    assert set(cited) == {4029, 4030, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040}
    assert cited[4032] == {
        "experiment_id": 4032,
        "sha256": hashlib.sha256(
            Path("results/experiment_4032_offarc_exec_verifier_transfer_collect.json").read_bytes()
        ).hexdigest(),
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["verifier_generalized_off_arc"].startswith("BARE BOOL")


def test_req_capstone_4041_clean_fixture_can_record_positive_measurements(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4041: clean upstream metrics drive all three argument fields."""

    payloads = _fixture_payloads()
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(), started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["verifier_generalized_off_arc"] is True
    assert artifact["g1_off_arc_transfer"]["outcome"] == "confirmed_ci_excludes_zero"
    assert artifact["search_layer_generalized"] is True
    assert artifact["g2_search_layer_generalization"]["nodes_expanded"] == 42
    assert artifact["decentralization_diagnosis"] == "latent"
    assert artifact["g3_decentralization_scaling"]["stronger_base_demo_perfect_coverage"] == 0.5
    assert artifact["total_games_solved"] == 8
    assert artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["target_game"] == "fixture-game"
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"]["action_savings_vs_cold"] == 22
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"]["v6_action_savings_vs_v5"] == 2
    assert artifact["accuracy_self_learning_hardware_deltas"]["hardware"]["kv260_latency_step_taken"] is True
    assert artifact["flagged_artifacts_skipped"] == []
    assert len(artifact["cited_upstream_artifacts"]) == len(mod.UPSTREAM_IDS)
    assert artifact["duration_s"] == 0.5


def test_req_capstone_4041_flagged_and_missing_inputs_cannot_satisfy_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4041: skipped or missing upstreams do not leak headline metrics."""

    payloads = _fixture_payloads()
    payloads[4032]["flagged_adversarial"] = True
    payloads[4035]["flagged_adversarial"] = True
    payloads[4037]["flagged_adversarial"] = True
    payloads.pop(4038)
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(payloads), returncodes={4032: 2, 4035: 2, 4037: 2}),
    )

    mod.validate_artifact(artifact)
    assert artifact["verifier_generalized_off_arc"] is False
    assert artifact["g1_off_arc_transfer"]["outcome"] == "skipped_flagged"
    assert artifact["search_layer_generalized"] is False
    assert artifact["g2_search_layer_generalization"]["nodes_expanded"] == 0
    assert artifact["decentralization_diagnosis"] == "flagged_skipped"
    assert artifact["total_games_solved"] == 6
    assert artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["game_solved"] is False
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 4038}]
    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [4032, 4035, 4037]
    assert {4032, 4035, 4037}.isdisjoint({row["experiment_id"] for row in artifact["cited_upstream_artifacts"]})


def test_req_capstone_4041_write_artifact_and_validate_rejects_regressions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4041: artifact writing validates required schema fields."""

    _write_default_artifacts(tmp_path, _fixture_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert written["schema"] == "carnot.capstone_v373_4041.v1"
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["verifier_generalized_off_arc"] = "true"
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["search_layer_generalized"] = 1
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["decentralization_diagnosis"] = "unknown"
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
    bad["cited_upstream_artifacts"] = [{"experiment_id": 4032, "sha256": "not-sha"}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["flagged_artifacts_skipped"] = {}
    with pytest.raises(ValueError, match="list"):
        mod.validate_artifact(bad)


def test_req_capstone_4041_summary_runner_uses_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4041: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / mod.DEFAULT_UPSTREAM_PATHS[4032]
    _write_json(path, _fixture_payloads()[4032])
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

    statuses = mod.summarize_existing_artifacts(tmp_path, {4032: path, 4038: None}, supplied=None)
    assert statuses == {4032: {"returncode": 0, "stdout": "summary", "stderr": ""}}
    assert mod.summarize_existing_artifacts(tmp_path, {4032: path}, supplied={4032: {"returncode": 2}}) == {
        4032: {"returncode": 2}
    }
    assert mod.list_float_metric({"ci": "not-a-list"}, "ci") == []
    assert mod.list_float_metric({"ci": [1, True, 2.5, "bad"]}, "ci") == [1.0, 2.5]
    assert mod.nested_int({"a": True}, ("a", "b")) == 0
    assert mod.off_arc_transfer_report(None, was_skipped=False)["outcome"] == "missing_or_blocked"
